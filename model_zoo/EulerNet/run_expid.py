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
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader
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
        result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), args['expid'], params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))

def run_inference(model, feature_map, params, args):
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
        
        # Debug: log what files were found
        if rank == 0:
            logging.info(f"Found {len(parquet_files)} parquet files, {len(csv_files)} csv files in {infer_data}")
            if parquet_files:
                logging.info(f"First 5 parquet files: {[os.path.basename(f) for f in parquet_files[:5]]}")
                logging.info(f"Last 5 parquet files: {[os.path.basename(f) for f in parquet_files[-5:]]}")
        
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
            # Debug: log sorted order
            if rank == 0:
                logging.info(f"Sorted files (first 10):")
                for i, f in enumerate(files[:10]):
                    num = extract_number(f)
                    logging.info(f"  [{i}] {os.path.basename(f)} -> index: {i}, number: {num}")
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
    
    # Clean output directory more thoroughly
    if rank == 0:
        import time
        max_retries = 5
        for retry in range(max_retries):
            try:
                # Check if another inference is already running
                if os.path.exists(lock_file):
                    # Check if lock is stale (older than 1 hour)
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 3600:  # 1 hour
                        logging.warning(f"Removing stale lock file (age: {lock_age:.0f}s)")
                        os.remove(lock_file)
                    else:
                        raise RuntimeError(f"Inference already running (lock file: {lock_file}, age: {lock_age:.0f}s)")
                
                if os.path.exists(output_dir):
                    # First remove all files
                    for root, dirs, files in os.walk(output_dir):
                        for f in files:
                            try:
                                os.remove(os.path.join(root, f))
                            except:
                                pass
                    # Then remove directory
                    import shutil
                    shutil.rmtree(output_dir)
                    time.sleep(2)  # Wait for filesystem
                
                # Create fresh directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Create lock file
                with open(lock_file, 'w') as f:
                    f.write(f"PID: {os.getpid()}, Time: {time.time()}")
                
                break
            except Exception as e:
                if retry == max_retries - 1:
                    logging.error(f"Failed to clean output directory after {max_retries} retries: {e}")
                    raise
                else:
                    logging.warning(f"Retry {retry + 1}/{max_retries} cleaning output directory: {e}")
                    time.sleep(3)
    
    if distributed and dist.is_initialized():
        distributed_barrier()
        
        # Double-check all ranks see the clean directory
        if rank == 0:
            logging.info(f"Output directory cleaned and locked: {output_dir}")
    
    if rank == 0:
        logging.info('******** Start Inference ********')
        logging.info(f"Results will be saved to directory: {output_dir}")
        # Debug: list files in output directory to ensure it's clean
        existing_files = glob.glob(os.path.join(output_dir, "*.parquet"))
        if existing_files:
            logging.warning(f"WARNING: Output directory not clean! Found {len(existing_files)} existing files")
            for f in existing_files[:5]:  # Show first 5 files
                logging.warning(f"  Found: {os.path.basename(f)}")
    
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
    
    # Distribute files across ranks for parallel processing
    # Create mapping of file to its original index
    rank_files = []
    file_indices = []  # Store original indices
    for i, f in enumerate(files):
        if i % world_size == rank:
            rank_files.append(f)
            file_indices.append(i)  # Save original index
    
    # Log distribution for debugging
    logging.info(f"Rank {rank}: Processing {len(rank_files)} files (indices: {file_indices})")
    
    # Debug: show which files will be processed
    if rank == 0 and rank_files:
        logging.info(f"Rank {rank} will process files:")
        for idx, f in zip(file_indices[:5], rank_files[:5]):  # Show first 5
            basename = os.path.basename(f)
            logging.info(f"  Index {idx}: {basename} -> will be saved as part_{idx}_rank{rank}.parquet")
        if len(rank_files) > 5:
            logging.info(f"  ... and {len(rank_files) - 5} more files")

    if rank_files:
        # Use zip to iterate with both file and original index
        for file_idx, f in zip(file_indices, tqdm(rank_files, desc=f"Inference (Rank {rank})", disable=rank!=0)):
            logger.setLevel(logging.WARNING)
            try:
                # Inference files usually have no label; skip labels
                ddf = feature_encoder.read_data(f, data_format=data_format, include_labels=False)

                # Extract IDs before preprocess (which filters columns)
                ids = ddf.select([c for c in ['phone', 'phone_md5'] if c in ddf.columns]).collect().to_pandas()

                # Preprocess (handles sequence conversion) and Transform
                df = feature_encoder.preprocess(ddf).collect().to_pandas()
                df = feature_encoder.transform(df)

                # Pass num_workers and other params to RankDataLoader
                # Use the same approach as in training: pass all params and let RankDataLoader handle them
                # We need to ensure test_data is passed correctly
                inference_params = params.copy()
                inference_params['test_data'] = [df]
                inference_params['data_loader'] = DataFrameDataLoader
                # For inference, we don't need shuffle
                inference_params['shuffle'] = False
                test_gen = RankDataLoader(feature_map, stage='test', **inference_params).make_iterator()

                model._verbose = 0
                current_batch_preds = model.predict(test_gen)

                if current_batch_preds is not None:
                    has_data = True

                    # Dict (Normal inference) or Array
                    pred_df = pd.DataFrame(current_batch_preds, columns=['pred'])
                    result_df = pd.concat([ids.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

                    # Save as Parquet part file with rank identifier
                    # Use original file index instead of rank_files index
                    part_file = os.path.join(output_dir, f"part_{file_idx}_rank{rank}.parquet")
                    result_df.to_parquet(part_file, index=False)
                
                model._verbose = params.get('verbose', 1)
                
                # Explicit garbage collection
                del ddf, df, ids, current_batch_preds
                gc.collect()

            finally:
                logger.setLevel(original_level)
    
    if distributed and dist.is_initialized():
        distributed_barrier()
    
    if rank == 0:
        if has_data:
            logging.info(f"Inference completed. Data saved in: {output_dir}")
            # Merge all rank files if distributed
            if distributed and world_size > 1:
                try:
                    merge_distributed_results(output_dir, world_size)
                except Exception as e:
                    logging.warning(f"Failed to merge distributed results: {e}")
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
