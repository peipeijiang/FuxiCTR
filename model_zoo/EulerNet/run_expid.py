# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(os.path.abspath("../.."))

# Suppress warnings from deprecated packages (pynvml/nvidia-ml-py)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader, ParquetTransformBlockDataLoader
from fuxictr.pytorch.torch_utils import seed_everything, init_distributed_env, distributed_barrier
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.distributed as dist
import signal
import threading


def finalize_files(file_writers):
    """Write all accumulated data to Parquet files (called when file completes or at the end)."""
    for file_idx, buffer_info in file_writers.items():
        if buffer_info['data']:
            # Concatenate all batches and write once
            final_df = pd.concat(buffer_info['data'], ignore_index=True)
            final_df.to_parquet(buffer_info['file'], index=False)


_stop_event = threading.Event()


def _install_signal_handlers():
    """Install SIGINT/SIGTERM handlers to allow graceful stop from UI/CLI."""
    def _handle(sig, frame):
        logging.warning(f"Received signal {sig}, requesting stop...")
        _stop_event.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            pass


def run_train(model, feature_map, params, args):
    rank = params.get('distributed_rank', 0)
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    if rank == 0:
        logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    test_result = {}
    if params["test_data"]:
        if rank == 0:
            logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    
    if rank == 0:
        result_filename = args['expid'] + '.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), args['expid'], params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))


def get_completed_files(output_dir):
    """扫描已完成的文件（part_N.parquet）"""
    import glob
    import re
    completed_files = set()
    if not os.path.exists(output_dir):
        return completed_files
    for f in glob.glob(os.path.join(output_dir, "part_*.parquet")):
        match = re.match(r'part_(\d+)\.parquet', os.path.basename(f))
        if match:
            completed_files.add(int(match.group(1)))
    return completed_files


def run_inference(model, feature_map, params, args):
    _install_signal_handlers()
    import torch.distributed as dist  # Import at function start for all ranks to use
    distributed = params.get('distributed', False)
    rank = params.get('distributed_rank', 0)
    world_size = params.get('distributed_world_size', 1)
    
    model.load_weights(model.checkpoint)
    
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_encoder = FeatureProcessor(**params).load_pickle(params.get('pickle_file', os.path.join(data_dir, "feature_processor.pkl")))
    feature_encoder.dtype_dict.update({'phone': str, 'phone_md5': str})

    infer_data = params['infer_data']
    if os.path.isdir(infer_data):
        # Get all parquet files
        parquet_files = glob.glob(os.path.join(infer_data, "*.parquet"))
        csv_files = glob.glob(os.path.join(infer_data, "*.csv"))
        
        # Sort files by numeric order (extract number from filename)
        def extract_number(filename):
            import re
            # Extract number from filename like "part_123.parquet" or "data_456.csv"
            basename = os.path.basename(filename)
            match = re.search(r'(\d+)', basename)
            return int(match.group(1)) if match else 0
        
        if parquet_files:
            files = sorted(parquet_files, key=extract_number)
            data_format = 'parquet'
        elif csv_files:
            files = sorted(csv_files, key=extract_number)
            data_format = 'csv'
        else:
            files = []
            data_format = 'parquet'  # default
            if rank == 0:
                logging.warning(f"No parquet or csv files found in {infer_data}")
    else:
        files = [infer_data]
        data_format = 'parquet' if infer_data.endswith('.parquet') else 'csv'
        if rank == 0:
            logging.info(f"Single file inference: {infer_data}")

    # Output directory setup (Parquet format for Big Data)
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")
    
    # Add lock file to prevent multiple inference processes
    lock_file = os.path.join(output_dir, ".inference_lock")
    
    # Clean output directory more thoroughly - ALL ranks participate
    import time
    max_retries = 5
    
    for retry in range(max_retries):
        try:
            # Only rank 0 handles lock file and directory creation
            if rank == 0:
                # Check if another inference is already running
                if os.path.exists(lock_file):
                    # Check if lock is stale (older than 5 minutes)
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 300:  # 5 minutes (降低阈值，便于快速恢复)
                        logging.warning(f"Removing stale lock file (age: {lock_age:.0f}s)")
                        os.remove(lock_file)
                    else:
                        # 检查锁文件中的PID是否还在运行
                        try:
                            with open(lock_file, 'r') as f:
                                content = f.read()
                                if 'PID:' in content:
                                    import re
                                    match = re.search(r'PID:\s*(\d+)', content)
                                    if match:
                                        pid = int(match.group(1))
                                        # 检查进程是否存在
                                        try:
                                            os.kill(pid, 0)  # 检查进程是否存在
                                            # 进程还在运行
                                            raise RuntimeError(f"Inference already running (PID: {pid}, lock file: {lock_file}, age: {lock_age:.0f}s)")
                                        except OSError:
                                            # 进程已结束，删除锁文件
                                            logging.warning(f"Process {pid} not found, removing stale lock file")
                                            os.remove(lock_file)
                        except Exception as e:
                            logging.warning(f"Error checking lock file: {e}, removing it")
                            os.remove(lock_file)
                        else:
                            raise RuntimeError(f"Inference already running (lock file: {lock_file}, age: {lock_age:.0f}s)")

                # Create directory if not exists (preserve existing files for resume)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    # Check for completed files (resume mode)
                    completed = get_completed_files(output_dir)
                    if completed:
                        logging.info(f"Resume mode: {len(completed)} files already completed: {sorted(completed)}")

                # Create lock file
                with open(lock_file, 'w') as f:
                    f.write(f"PID: {os.getpid()}, Time: {time.time()}")

            # ALL ranks wait for directory to be ready
            if distributed and dist.is_initialized():
                distributed_barrier()

            # Double-check directory exists for all ranks
            if not os.path.exists(output_dir):
                # Try to create it if it doesn't exist (should only happen if rank 0 failed)
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except:
                    pass

            # Verify directory exists
            if not os.path.exists(output_dir):
                raise RuntimeError(f"Output directory does not exist: {output_dir}")

            if rank == 0:
                logging.info(f"Output directory ready: {output_dir}")
            
            break
        except Exception as e:
            if retry == max_retries - 1:
                logging.error(f"Failed to clean output directory after {max_retries} retries: {e}")
                raise
            else:
                if rank == 0:
                    logging.warning(f"Retry {retry + 1}/{max_retries} cleaning output directory: {e}")
                time.sleep(3)
    
    # Final barrier to ensure all ranks are ready
    if distributed and dist.is_initialized():
        distributed_barrier()
    
    if rank == 0:
        logging.info('******** Start Inference ********')
        logging.info(f"Results will be saved to directory: {output_dir}")
    
    import warnings
    from tqdm import tqdm
    warnings.simplefilter("ignore")
    logger = logging.getLogger()
    original_level = logger.level

    has_data = False

    # Log file distribution info
    if rank == 0:
        logging.info(f"Total files to process: {len(files)}")
        logging.info(f"World size: {world_size}, Rank: {rank}")

    # Get completed files for resume mode
    completed_files = get_completed_files(output_dir) if rank == 0 else set()

    # Broadcast completed files to all ranks (for distributed inference)
    if distributed and dist.is_initialized():
        completed_list = list(completed_files)
        shared = [completed_list]
        dist.broadcast_object_list(shared, src=0)
        completed_files = set(shared[0])

    # Distribute files across ranks for parallel processing
    # Create mapping of file to its original index
    rank_files = []
    file_indices = []  # Store original indices

    for i, f in enumerate(files):
        # Skip completed files (resume mode)
        if i in completed_files:
            continue
        if i % world_size == rank:
            rank_files.append(f)
            file_indices.append(i)  # Save original index

    # Log distribution for debugging
    if completed_files:
        logging.info(f"Rank {rank}: Skipping {len(completed_files)} completed files: {sorted(completed_files)}")
    logging.info(f"Rank {rank}: Processing {len(rank_files)} files (indices: {file_indices})")

    # Create mapping from block index (in rank_files) to original file index
    block_idx_to_file_idx = {block_idx: file_idx for block_idx, file_idx in enumerate(file_indices)}

    # Log num_workers setting before starting inference (before logger.setLevel)
    if rank == 0:
        num_workers_value = params.get('num_workers', 1)
        batch_size_value = params.get('batch_size', 10000)
        chunk_size_value = params.get('infer_chunk_size', 10000)
        data_loader_type = "ParquetTransformBlockDataLoader (streaming)"

        # 检测 DDP 状态
        is_ddp = distributed and dist.is_initialized()

        logging.info(f"=" * 60)
        logging.info(f"Rank {rank}: Inference Configuration:")
        logging.info(f"  - DataLoader: {data_loader_type}")
        logging.info(f"  - Distributed: {distributed}")
        logging.info(f"  - DDP initialized: {is_ddp}")
        logging.info(f"  - Num Workers: {num_workers_value}")
        if is_ddp and num_workers_value > 0:
            logging.warning(
                f"  ⚠️  DDP + num_workers={num_workers_value} may cause hangs. "
                f"DataLoader will auto-adjust to num_workers=0"
            )
        logging.info(f"  - Batch Size: {batch_size_value}")
        logging.info(f"  - Chunk Size: {chunk_size_value} (rows per chunk for memory optimization)")
        logging.info(f"  - Files to process: {len(rank_files)}")
        logging.info(f"  - Streaming mode: Enabled (batch-level incremental write)")
        logging.info(f"=" * 60)

    # Ensure all ranks log their configuration
    logging.info(f"Rank {rank}: Using num_workers={params.get('num_workers', 1)} for inference")

    if rank_files:
        # Scan all files to get total row counts (for detecting file completion)
        import pyarrow.parquet as pq
        file_total_rows = {}
        # Use block_idx as key (matching _file_idx from DataPipe)
        for block_idx, file_path in enumerate(rank_files):
            try:
                parquet_file = pq.ParquetFile(file_path)
                file_total_rows[block_idx] = parquet_file.metadata.num_rows
            except Exception as e:
                logging.warning(f"Failed to read metadata for {file_path}: {e}")
                file_total_rows[block_idx] = 0

        if rank == 0:
            logging.info(f"Rank {rank}: Scanned {len(file_total_rows)} files, total rows: {sum(file_total_rows.values())}")

        # Use streaming ParquetTransformBlockDataLoader for better memory efficiency
        # Pass all rank_files at once to leverage multi-worker data loading
        inference_params = params.copy()
        inference_params['test_data'] = rank_files  # Pass file paths directly
        inference_params['data_loader'] = ParquetTransformBlockDataLoader  # Streaming loader
        inference_params['feature_encoder'] = feature_encoder
        inference_params['id_cols'] = ['phone', 'phone_md5']
        inference_params['shuffle'] = False
        inference_params['batch_size'] = params.get('batch_size', 10000)
        inference_params['num_workers'] = params.get('num_workers', 0)
        inference_params['multiprocessing_context'] = params.get('multiprocessing_context', 'fork')

        inference_params['chunk_size'] = params.get('infer_chunk_size', 10000)  # Default: 10K rows per chunk for memory efficiency

        test_gen = RankDataLoader(feature_map, stage='test', **inference_params).make_iterator()

        model._verbose = 0

        # Track processed rows per file to detect completion
        file_processed_rows = {}
        # Track last logged progress percentage per file to avoid log spam
        file_progress_logged = {}
        # State for single-line file progress (rank0 + TTY)
        file_progress_line_len = 0
        file_progress_current = None
        # Accumulate predictions directly in file_writers (no batch_buffer)
        file_writers = {}

        try:
            is_tty = sys.stdout.isatty()
            tqdm_disable = rank != 0 or not is_tty
            for batch_data in tqdm(test_gen,
                                   desc=f"Inference (Rank {rank})",
                                   disable=tqdm_disable,
                                   dynamic_ncols=True):
                if _stop_event.is_set():
                    logging.warning(f"Rank {rank}: Stop requested, exiting inference loop")
                    break
                # Extract file index and IDs from batch
                file_idx_arr = batch_data.pop("_file_idx", None)
                if file_idx_arr is None:
                    logging.warning("No _file_idx in batch, skipping")
                    continue

                # Get unique files in this batch
                file_indices_in_batch = np.unique(file_idx_arr)

                # Extract ID columns
                ids_batch = {}
                for col in ['phone', 'phone_md5']:
                    if col in batch_data:
                        ids_batch[col] = batch_data.pop(col)
                    else:
                        ids_batch[col] = None

                # Predict
                with torch.no_grad():
                    pred_dict = model.forward(batch_data)
                    batch_preds = pred_dict["y_pred"].data.cpu().numpy().reshape(-1)

                # Split predictions by file and accumulate directly to file_writers
                for block_idx in file_indices_in_batch:
                    # Convert block_idx to original file index
                    original_file_idx = block_idx_to_file_idx.get(block_idx, block_idx)
                    
                    # Find rows belonging to this file
                    mask = file_idx_arr == block_idx
                    f_preds = batch_preds[mask]
                    f_count = len(f_preds)

                    # Get IDs for this file
                    f_ids = {}
                    for col in ['phone', 'phone_md5']:
                        if ids_batch[col] is not None:
                            if torch.is_tensor(ids_batch[col]):
                                f_ids[col] = ids_batch[col][mask].cpu().numpy()
                            else:
                                f_ids[col] = ids_batch[col][mask]

                    # Create DataFrame for this batch's predictions
                    result_df = pd.DataFrame({'pred': f_preds})
                    for col, vals in f_ids.items():
                        result_df[col] = vals

                    # Accumulate directly to file_writers (no intermediate batch_buffer)
                    part_file = os.path.join(output_dir, f"part_{original_file_idx}_rank{rank}.parquet")
                    if original_file_idx in file_writers:
                        file_writers[original_file_idx]['data'].append(result_df)
                        file_writers[original_file_idx]['count'] += f_count
                    else:
                        file_writers[original_file_idx] = {
                            'data': [result_df],
                            'count': f_count,
                            'file': part_file
                        }

                    # Track processed rows
                    file_processed_rows[block_idx] = file_processed_rows.get(block_idx, 0) + f_count

                    # Progress logging per file (only when metadata is available)
                    total_rows = file_total_rows.get(block_idx, 0)
                    if total_rows > 0:
                        progress_pct = min(100, int(file_processed_rows[block_idx] * 100 / total_rows))
                        last_pct = file_progress_logged.get(block_idx, -1)

                        # Rank0 + TTY: update a single-line file progress to avoid multiple lines
                        if rank == 0 and is_tty:
                            msg = f"\r[Rank {rank}] part_{original_file_idx} {progress_pct}% ({file_processed_rows[block_idx]}/{total_rows} rows)"
                            pad = max(0, file_progress_line_len - len(msg))
                            sys.stdout.write(msg + (" " * pad))
                            sys.stdout.flush()
                            file_progress_line_len = len(msg)
                            file_progress_current = original_file_idx
                            # Only log at completion to keep log clean
                            if progress_pct == 100 and last_pct < 100:
                                logging.info(
                                    f"Rank {rank}: File part_{original_file_idx} progress {progress_pct}% "
                                    f"({file_processed_rows[block_idx]}/{total_rows} rows)"
                                )
                                file_progress_logged[block_idx] = progress_pct
                        else:
                            # Non-TTY or non-rank0: log every 10% or on completion
                            if progress_pct >= last_pct + 10 or progress_pct == 100:
                                logging.info(
                                    f"Rank {rank}: File part_{original_file_idx} progress {progress_pct}% "
                                    f"({file_processed_rows[block_idx]}/{total_rows} rows)"
                                )
                                file_progress_logged[block_idx] = progress_pct

                    # Check if file is complete and write immediately
                    # Only check if we have valid row count (> 0)
                    if file_total_rows.get(block_idx, 0) > 0 and file_processed_rows[block_idx] >= file_total_rows[block_idx]:
                        # File complete: write and free memory
                        finalize_files({original_file_idx: file_writers[original_file_idx]})
                        del file_writers[original_file_idx]  # Free memory immediately

                        # Immediately finalize this file (simple rename, no merge needed)
                        # Each file is processed by only one rank, so just rename temp → final
                        temp_file = os.path.join(output_dir, f"part_{original_file_idx}_rank{rank}.parquet")
                        final_file = os.path.join(output_dir, f"part_{original_file_idx}.parquet")

                        if os.path.exists(temp_file):
                            os.rename(temp_file, final_file)
                            logging.info(f"Rank {rank}: Finalized part_{original_file_idx}.parquet")
                            # Finish the single-line display for this file
                            if rank == 0 and is_tty and file_progress_current == original_file_idx:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                                file_progress_line_len = 0
                                file_progress_current = None

                        # Synchronize all ranks after each file completion
                        if distributed and dist.is_initialized():
                            distributed_barrier()

                has_data = True

        finally:
            # Finalize any remaining files (should be empty if all files completed)
            finalize_files(file_writers)
            file_writers.clear()

            logger.setLevel(original_level)

    if distributed and dist.is_initialized() and not _stop_event.is_set():
        distributed_barrier()
    
    if rank == 0:
        if has_data:
            logging.info(f"Inference completed. Data saved in: {output_dir}")
            # Clean up any remaining temp files (should be empty if all finalized immediately)
            remaining_temp_files = glob.glob(os.path.join(output_dir, "part_*_rank*.parquet"))
            if remaining_temp_files:
                logging.warning(f"Found {len(remaining_temp_files)} unfinalized temp files, merging...")
                try:
                    merge_distributed_results(output_dir, world_size)
                except Exception as e:
                    logging.warning(f"Failed to merge remaining results: {e}")
            else:
                logging.info("All files finalized successfully during inference")
        else:
            logging.warning("No data found in infer_data!")
        
        # Clean up lock file
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logging.info(f"Lock file removed: {lock_file}")
        except Exception as e:
            logging.warning(f"Failed to remove lock file: {e}")


def merge_distributed_results(output_dir, world_size):
    """Merge parquet files from all ranks into single files."""
    import pandas as pd
    import glob
    
    # Find all part files
    part_files = glob.glob(os.path.join(output_dir, "part_*_rank*.parquet"))
    if not part_files:
        return
    
    # Group by part number (without rank)
    part_groups = {}
    for f in part_files:
        basename = os.path.basename(f)
        # Extract part number (e.g., "part_0" from "part_0_rank0.parquet")
        import re
        match = re.match(r'part_(\d+)_rank\d+\.parquet', basename)
        if match:
            part_num = int(match.group(1))
            if part_num not in part_groups:
                part_groups[part_num] = []
            part_groups[part_num].append(f)
    
    # Merge each group
    for part_num, files in sorted(part_groups.items()):
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
                os.remove(f)  # Remove individual rank file
            except Exception as e:
                logging.warning(f"Failed to read or remove {f}: {e}")
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_file = os.path.join(output_dir, f"part_{part_num}.parquet")
            merged_df.to_parquet(merged_file, index=False)
            logging.info(f"Merged {len(files)} rank files into {merged_file}")


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id} --mode {train|inference}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--distributed', action='store_true', help='Enable torch.distributed for multi-GPU training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Torch distributed backend to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='The running mode.')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    distributed, rank, world_size, local_rank = init_distributed_env(args)
    params['gpu'] = local_rank if distributed else args['gpu']
    params['distributed'] = distributed
    params['distributed_rank'] = rank
    params['distributed_world_size'] = world_size
    params['local_rank'] = local_rank
    if rank == 0:
        set_logger(params)
    else:
        logging.basicConfig(level=logging.INFO if params.get('verbose', 0) > 0 else logging.WARNING)
    logging.info("Rank {} initialized (world size {}).".format(rank, world_size))
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] in ["csv", "parquet"] and args['mode'] == 'train':
        data_splits = (None, None, None)
        if rank == 0:
            feature_encoder = FeatureProcessor(**params)
            data_splits = build_dataset(feature_encoder, **params)
        if distributed and dist.is_initialized():
            shared = [data_splits]
            dist.broadcast_object_list(shared, src=0)
            data_splits = shared[0]
        params["train_data"], params["valid_data"], params["test_data"] = data_splits
        if distributed and dist.is_initialized():
            distributed_barrier()
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    if rank == 0:
        logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    if args['mode'] == 'train':
        run_train(model, feature_map, params, args)
    elif args['mode'] == 'inference':
        run_inference(model, feature_map, params, args)

    if distributed and dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()
