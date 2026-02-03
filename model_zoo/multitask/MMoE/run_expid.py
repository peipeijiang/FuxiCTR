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

"""
Universal run_expid.py - Generic inference script for all model types

Features:
- Auto-detects project root (works at any depth)
- Supports single-task models
- Supports multi-task models (MMoE, PLE, etc.)
- Supports sweep mode (multi-scenario inference)
- Handles binary_classification_logits correctly
- Streaming inference with ParquetTransformBlockDataLoader
- Distributed inference support

Usage:
    # Copy this file to your model directory:
    # - For single-task: model_zoo/YourModel/run_expid.py
    # - For multi-task: model_zoo/multitask/YourModel/run_expid.py

    python run_expid.py --expid your_experiment --mode inference --gpu 0
"""

import os
import sys

# Auto-detect project root (works at any directory depth)
current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)
root_path = os.path.abspath(current_path)
for _ in range(5):  # Scan up to 5 levels
    if os.path.exists(os.path.join(root_path, "fuxictr")):
        break
    root_path = os.path.dirname(root_path)
sys.path.append(root_path)  # Add project root
sys.path.append(os.path.join(root_path, "model_zoo/common"))  # Add common directory

# Suppress warnings from deprecated packages
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything, init_distributed_env, distributed_barrier
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader, ParquetTransformBlockDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.inference import SweepInference, ParquetWriterWrapper, Inferenceutils
import src as model_zoo
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
import time
import re
import pyarrow.parquet as pq


_stop_event = threading.Event()


def _install_signal_handlers():
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    def _handle(sig, frame):
        logging.warning(f"Received signal {sig}, requesting stop...")
        _stop_event.set()
        # Clean up task state on signal
        try:
            cleanup_task_state()
            cleanup_pytorch_tmp_files()
        except:
            pass
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            pass


def cleanup_task_state(pid=None):
    """Clean up dashboard task state file for this process."""
    if pid is None:
        pid = os.getpid()

    # Try to find and remove task state file
    task_state_dir = os.path.join(root_path, "dashboard", "state", "tasks")
    if not os.path.exists(task_state_dir):
        return

    try:
        for f in os.listdir(task_state_dir):
            if f.endswith(".json") and f"_{pid}.json" in f:
                fpath = os.path.join(task_state_dir, f)
                os.remove(fpath)
                logging.info(f"Cleaned up task state: {f}")
                break
    except Exception as e:
        logging.warning(f"Failed to cleanup task state: {e}")


def cleanup_pytorch_tmp_files():
    """Clean up PyTorch DDP/RPC temporary files."""
    import glob
    try:
        # Find tmp directories in model directory
        pattern = os.path.join(root_path, "model_zoo", "**", "tmp*", "_remote_module_non_scriptable.py")
        matching_files = glob.glob(pattern, recursive=True)

        for tmp_file in matching_files:
            try:
                # Remove the file and its parent tmp directory
                tmp_dir = os.path.dirname(tmp_file)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                if os.path.exists(tmp_dir) and tmp_dir.startswith(os.path.join(root_path, "model_zoo")):
                    # Only remove if it's empty or only contains the remote_module file
                    try:
                        os.rmdir(tmp_dir)
                        logging.info(f"Cleaned up PyTorch tmp dir: {tmp_dir}")
                    except OSError:
                        # Directory not empty, just remove the file
                        logging.info(f"Cleaned up PyTorch tmp file: {tmp_file}")
            except Exception as e:
                logging.warning(f"Failed to cleanup {tmp_file}: {e}")
    except Exception as e:
        logging.warning(f"Failed to cleanup PyTorch tmp files: {e}")


def run_train(model, feature_map, params, args, workflow_logger=None):
    \"\"\"Training function.

    Args:
        model: Model instance for training
        feature_map: Feature map for data processing
        params: Parameters dictionary
        args: Arguments dictionary
        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting
    \"\"\"
    """Training function."""
    rank = params.get('distributed_rank', 0)
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    # Set workflow_logger on model if available
    if workflow_logger:
        model._workflow_logger = workflow_logger
    model.fit(train_gen, validation_data=valid_gen, **params)

    if rank == 0:
        logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    if params.get("test_data"):
        if rank == 0:
            logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    if rank == 0:
        result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                        ' '.join(sys.argv), args['expid'], params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))


def get_completed_files(output_dir):
    """Get list of completed files (part_N.parquet)."""
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


def merge_distributed_results(output_dir, world_size):
    """Merge parquet files from all ranks into single files."""
    part_files = glob.glob(os.path.join(output_dir, "part_*_rank*.parquet"))
    if not part_files:
        return

    part_groups = {}
    for f in part_files:
        basename = os.path.basename(f)
        match = re.match(r'part_(\d+)_rank\d+\.parquet', basename)
        if match:
            part_num = int(match.group(1))
            if part_num not in part_groups:
                part_groups[part_num] = []
            part_groups[part_num].append(f)

    for part_num, files in sorted(part_groups.items()):
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to read or remove {f}: {e}")

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_file = os.path.join(output_dir, f"part_{part_num}.parquet")
            merged_df.to_parquet(merged_file, index=False)
            logging.info(f"Merged {len(files)} rank files into {merged_file}")


def run_inference(model, feature_map, params, args, workflow_logger=None):
    \"\"\"Inference function.

    Args:
        model: Model instance for inference
        feature_map: Feature map for data processing
        params: Parameters dictionary
        args: Arguments dictionary
        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting
    \"\"\"
    """Universal inference function supporting single-task, multi-task, and sweep modes."""
    _install_signal_handlers()
    distributed = params.get('distributed', False)
    rank = params.get('distributed_rank', 0)
    world_size = params.get('distributed_world_size', 1)

    model.load_weights(model.checkpoint)

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_encoder = FeatureProcessor(**params).load_pickle(
        params.get('pickle_file', os.path.join(data_dir, "feature_processor.pkl"))
    )
    feature_encoder.dtype_dict.update({'phone': str, 'phone_md5': str})

    # Setup input data files
    infer_data = params['infer_data']
    if os.path.isdir(infer_data):
        parquet_files = glob.glob(os.path.join(infer_data, "*.parquet"))
        csv_files = glob.glob(os.path.join(infer_data, "*.csv"))

        def extract_number(filename):
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
            data_format = 'parquet'
            if rank == 0:
                logging.warning(f"No parquet or csv files found in {infer_data}")
    else:
        files = [infer_data]
        data_format = 'parquet' if infer_data.endswith('.parquet') else 'csv'
        if rank == 0:
            logging.info(f"Single file inference: {infer_data}")

    # Setup output directory
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")
    lock_file = os.path.join(output_dir, ".inference_lock")

    # Initialize SweepInference (auto-detects sweep mode)
    sweep_inference = SweepInference(model, feature_map, params)
    sweep_inference.set_id_to_token(feature_encoder)

    if rank == 0:
        mode_str = "SWEEP" if sweep_inference.sweep_enabled else "NORMAL"
        logging.info(f"Inference Mode: {mode_str}")
        if sweep_inference.sweep_enabled:
            logging.info(f"Sweep column: {sweep_inference.sweep_col}, Domains per pass: {sweep_inference.domains_per_pass}")

    # Clean output directory and setup lock file
    max_retries = 5
    for retry in range(max_retries):
        try:
            if rank == 0:
                if os.path.exists(lock_file):
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 300:
                        logging.warning(f"Removing stale lock file (age: {lock_age:.0f}s)")
                        os.remove(lock_file)
                    else:
                        try:
                            with open(lock_file, 'r') as f:
                                content = f.read()
                                if 'PID:' in content:
                                    match = re.search(r'PID:\s*(\d+)', content)
                                    if match:
                                        pid = int(match.group(1))
                                        try:
                                            os.kill(pid, 0)
                                            raise RuntimeError(f"Inference already running (PID: {pid})")
                                        except OSError:
                                            logging.warning(f"Process {pid} not found, removing stale lock file")
                                            os.remove(lock_file)
                        except Exception:
                            os.remove(lock_file)
                        else:
                            raise RuntimeError(f"Inference already running (age: {lock_age:.0f}s)")

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    completed = get_completed_files(output_dir)
                    if completed:
                        logging.info(f"Resume mode: {len(completed)} files already completed: {sorted(completed)}")

                with open(lock_file, 'w') as f:
                    f.write(f"PID: {os.getpid()}, Time: {time.time()}")

            if distributed and dist.is_initialized():
                distributed_barrier()

            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except:
                    pass

            if not os.path.exists(output_dir):
                raise RuntimeError(f"Output directory does not exist: {output_dir}")

            if rank == 0:
                logging.info(f"Output directory ready: {output_dir}")

            break
        except Exception as e:
            if retry == max_retries - 1:
                logging.error(f"Failed to setup output directory after {max_retries} retries: {e}")
                raise
            else:
                if rank == 0:
                    logging.warning(f"Retry {retry + 1}/{max_retries}: {e}")
                time.sleep(3)

    if distributed and dist.is_initialized():
        distributed_barrier()

    if rank == 0:
        logging.info('******** Start Inference ********')
        logging.info(f"Results will be saved to: {output_dir}")

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

    # Broadcast completed files to all ranks
    if distributed and dist.is_initialized():
        completed_list = list(completed_files)
        shared = [completed_list]
        dist.broadcast_object_list(shared, src=0)
        completed_files = set(shared[0])

    # Distribute files across ranks
    rank_files = []
    file_indices = []

    for i, f in enumerate(files):
        if i not in completed_files:
            if i % world_size == rank:
                rank_files.append(f)
                file_indices.append(i)

    if completed_files:
        logging.info(f"Rank {rank}: Skipping {len(completed_files)} completed files")
    logging.info(f"Rank {rank}: Processing {len(rank_files)} files (indices: {file_indices})")

    # Create mapping from block index to original file index
    block_idx_to_file_idx = {block_idx: file_idx for block_idx, file_idx in enumerate(file_indices)}

    # Log configuration
    if rank == 0:
        num_workers_value = params.get('num_workers', 1)
        batch_size_value = params.get('batch_size', 10000)
        chunk_size_value = params.get('infer_chunk_size', 10000)

        is_ddp = distributed and dist.is_initialized()

        logging.info(f"=" * 60)
        logging.info(f"Rank {rank}: Inference Configuration:")
        logging.info(f"  - Distributed: {distributed}")
        logging.info(f"  - DDP initialized: {is_ddp}")
        logging.info(f"  - Num Workers: {num_workers_value}")
        if is_ddp and num_workers_value > 0:
            logging.warning(f"  ⚠️  DDP + num_workers={num_workers_value} may cause hangs")
        logging.info(f"  - Batch Size: {batch_size_value}")
        logging.info(f"  - Chunk Size: {chunk_size_value}")
        logging.info(f"  - Files to process: {len(rank_files)}")
        logging.info(f"=" * 60)

    logging.info(f"Rank {rank}: Using num_workers={params.get('num_workers', 1)} for inference")

    if rank_files:
        # Scan files for total row counts
        file_total_rows = {}
        for block_idx, file_path in enumerate(rank_files):
            try:
                parquet_file = pq.ParquetFile(file_path)
                file_total_rows[block_idx] = parquet_file.metadata.num_rows
            except Exception as e:
                logging.warning(f"Failed to read metadata for {file_path}: {e}")
                file_total_rows[block_idx] = 0

        if rank == 0:
            logging.info(f"Rank {rank}: Scanned {len(file_total_rows)} files, total rows: {sum(file_total_rows.values())}")

        # Setup DataLoader
        inference_params = params.copy()
        inference_params['test_data'] = rank_files
        inference_params['data_loader'] = ParquetTransformBlockDataLoader
        inference_params['feature_encoder'] = feature_encoder
        inference_params['id_cols'] = ['phone', 'phone_md5']
        inference_params['shuffle'] = False
        inference_params['batch_size'] = params.get('batch_size', 10000)
        inference_params['num_workers'] = params.get('num_workers', 0)
        inference_params['multiprocessing_context'] = params.get('multiprocessing_context', 'fork')
        inference_params['chunk_size'] = params.get('infer_chunk_size', 10000)

        test_gen = RankDataLoader(feature_map, stage='test', **inference_params).make_iterator()

        model._verbose = 0

        # Track progress
        file_processed_rows = {}
        file_progress_logged = {}
        file_progress_line_len = 0
        file_progress_current = None

        # Initialize ParquetWriterWrapper
        writer_wrapper = ParquetWriterWrapper(
            output_dir,
            buffer_limit=params.get('sweep_write_buffer_rows', 50000),
            filename_fmt="part_{fid}_rank" + str(rank) + ".parquet"
        )

        try:
            from tqdm import tqdm

            is_tty = sys.stdout.isatty()

            # Setup dual progress bars
            total_pbar = None
            file_pbar = None
            batch_pbar = None

            if is_tty:
                # Total progress bar (position 0)
                total_pbar = tqdm(
                    total=len(file_total_rows),
                    desc=f"Rank {rank}: Total",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    disable=False
                )

                # Current file progress bar (position 1)
                file_pbar = tqdm(
                    total=100,
                    desc=f"Rank {rank}: Current file",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    disable=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}% [{elapsed}<{remaining}]"
                )

            # Track current file for progress update
            current_file_id = None
            current_file_total = 0
            current_file_processed = 0

            # Track last logged progress for dashboard
            last_logged_files_done = -1
            batch_idx = 0

            for batch_data in test_gen:
                if _stop_event.is_set():
                    logging.warning(f"Rank {rank}: Stop requested, exiting inference loop")
                    break

                # Extract meta info
                file_idx_arr = batch_data.pop("_file_idx")
                ids_batch = {c: batch_data.pop(c) for c in list(batch_data.keys()) if c in ['phone', 'phone_md5']}

                # Map block indices to original file indices
                original_file_indices = []
                for b_idx in file_idx_arr:
                    if isinstance(b_idx, torch.Tensor):
                        b_idx = b_idx.item()
                    original_file_indices.append(block_idx_to_file_idx.get(b_idx, b_idx))

                original_file_indices = np.array(original_file_indices)
                unique_files = np.unique(original_file_indices)

                # Prepare ID cache
                id_cache = Inferenceutils._prepare_id_cache(ids_batch, unique_files, original_file_indices)

                # Run batch via SweepInference
                batch_success = sweep_inference.run_batch(batch_data, unique_files, id_cache, writer_wrapper)

                if batch_success:
                    has_data = True

                # Update progress tracking
                files_completed_this_batch = []
                for b_idx in file_idx_arr:
                    if isinstance(b_idx, torch.Tensor):
                        b_idx = b_idx.item()
                    file_processed_rows[b_idx] = file_processed_rows.get(b_idx, 0) + 1

                    # Check if file is completed
                    total_rows = file_total_rows.get(b_idx, 0)
                    if total_rows > 0 and file_processed_rows[b_idx] >= total_rows:
                        if b_idx not in files_completed_this_batch:
                            files_completed_this_batch.append(b_idx)

                # Count total files completed
                files_done_count = sum(1 for idx, rows in file_processed_rows.items()
                                      if file_total_rows.get(idx, 0) > 0 and rows >= file_total_rows[idx])

                # Update progress bars
                if is_tty and total_pbar:
                    total_pbar.n = files_done_count
                    total_pbar.refresh()

                    # Update current file progress
                    if len(unique_files) > 0:
                        # Get the file being processed (first one in unique_files)
                        current_fid = unique_files[0]
                        if current_fid != current_file_id:
                            current_file_id = current_fid
                            current_file_total = file_total_rows.get(current_fid, 0)
                            current_file_processed = 0

                        current_file_processed = file_processed_rows.get(current_fid, 0)
                        if current_file_total > 0:
                            pct = min(100, int(current_file_processed * 100 / current_file_total))
                            file_pbar.n = pct
                            original_fid = block_idx_to_file_idx.get(current_fid, current_fid)
                            file_pbar.set_description(f"Rank {rank}: part_{original_fid}")
                            file_pbar.refresh()

                # Log progress for dashboard (every file completion or every 50 batches)
                batch_idx += 1
                if rank == 0 and (files_done_count > last_logged_files_done or batch_idx % 50 == 0):
                    last_logged_files_done = files_done_count
                    pct = int(files_done_count * 100 / len(file_total_rows)) if len(file_total_rows) > 0 else 0
                    logging.info(f"Progress: {files_done_count}/{len(file_total_rows)} files completed ({pct}%)")

                del batch_data

            # Close progress bars
            if is_tty:
                if total_pbar:
                    total_pbar.close()
                if file_pbar:
                    file_pbar.close()

        finally:
            writer_wrapper.close()
            logger.setLevel(original_level)

    # 立即重命名：每个rank完成处理的文件直接重命名为最终文件名
    # 这样可以及时释放内存，不需要等到最后的全局merge
    finalize_count = 0
    temp_files = glob.glob(os.path.join(output_dir, f"part_*_rank{rank}.parquet"))
    for temp_file in temp_files:
        try:
            basename = os.path.basename(temp_file)
            # 从 part_0_rank0.parquet 提取 part_0
            match = re.match(r'part_(\d+)_rank\d+\.parquet', basename)
            if match:
                part_num = match.group(1)
                final_file = os.path.join(output_dir, f"part_{part_num}.parquet")
                # 检查是否已存在（可能被其他rank重命名了）
                if not os.path.exists(final_file):
                    os.rename(temp_file, final_file)
                    finalize_count += 1
                    if rank == 0:  # 只打印rank 0的日志
                        logging.info(f"Finalized part_{part_num}.parquet")
                else:
                    # 文件已存在，删除临时文件
                    os.remove(temp_file)
        except Exception as e:
            logging.warning(f"Rank {rank}: Failed to finalize {temp_file}: {e}")

    if finalize_count > 0 and rank == 0:
        logging.info(f"Rank {rank}: Finalized {finalize_count} files")

    if distributed and dist.is_initialized() and not _stop_event.is_set():
        distributed_barrier()

    if rank == 0:
        if has_data:
            logging.info(f"Inference completed. Data saved in: {output_dir}")
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

        # Clean up dashboard task state
        try:
            cleanup_task_state()
        except Exception as e:
            logging.warning(f"Failed to cleanup task state: {e}")

        # Clean up PyTorch temporary files
        try:
            cleanup_pytorch_tmp_files()
        except Exception as e:
            logging.warning(f"Failed to cleanup PyTorch tmp files: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='The running mode.')
    parser.add_argument('--distributed', action='store_true', help='Enable torch.distributed for multi-GPU training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Torch distributed backend to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
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

    # Load feature_map
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "parquet" and args['mode'] == 'train':
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

    # Load model
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()

    # Initialize workflow_logger if in Dashboard mode
    workflow_logger = None
    if os.environ.get('FUXICTR_WORKFLOW_MODE') == 'dashboard':
        try:
            from fuxictr.workflow.utils.logger import get_workflow_logger
            task_id = os.environ.get('FUXICTR_TASK_ID')
            if task_id:
                workflow_logger = get_workflow_logger(int(task_id))
                if rank == 0:
                    logging.info(f\"Workflow logger initialized for task {task_id}\")
        except Exception as e:
            logging.warning(f\"Failed to initialize workflow logger: {e}\")

    if args['mode'] == 'train':
        run_train(model, feature_map, params, args, workflow_logger=workflow_logger)
    elif args['mode'] == 'inference':
        run_inference(model, feature_map, params, args, workflow_logger=workflow_logger)

    if distributed and dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()

    # Clean up task state file on successful completion
    if rank == 0:
        try:
            cleanup_task_state()
            cleanup_pytorch_tmp_files()
        except Exception as e:
            logging.warning(f"Failed to cleanup on completion: {e}")
