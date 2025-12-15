import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(os.path.abspath("../../.."))
sys.path.append(os.path.abspath("../common")) # Add common directory to path
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything, init_distributed_env
import torch.multiprocessing as mp
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import glob
from pathlib import Path
from sweep_inference import run_inference_files
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
    if params.get("test_data"):
        if rank == 0:
            logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        if test_gen:
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
    if distributed and rank != 0:
        logging.info("Rank %d skips inference; only master runs inference mode.", rank)
        return
    model.load_weights(model.checkpoint)
    
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_encoder = FeatureProcessor(**params).load_pickle(params.get('pickle_file', os.path.join(data_dir, "feature_processor.pkl")))
    feature_encoder.dtype_dict.update({'phone': str, 'phone_md5': str})

    infer_data = params['infer_data']
    if os.path.isdir(infer_data):
        files = sorted(glob.glob(os.path.join(infer_data, "*.parquet"))) or sorted(glob.glob(os.path.join(infer_data, "*.csv")))
    else:
        files = [infer_data]

    # Output directory setup (Parquet format for Big Data)
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('******** Start Inference ********')
    logging.info(f"Results will be saved to directory: {output_dir}")
    has_data = run_inference_files(model, feature_map, feature_encoder, params, args, files, output_dir)

    if has_data:
        logging.info(f"Inference completed. Data saved in: {output_dir}")
    else:
        logging.warning("No data found in infer_data!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='APG_AITM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--distributed', action='store_true', help='Enable torch.distributed for multi-GPU training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Torch distributed backend to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='The running mode.')
    parser.add_argument('--sweep', action='store_true', help='Run sweep inference (iterate over domain feature).')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    mp.set_start_method("spawn", force=True)
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

    # Load feature_map from json
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
            dist.barrier()
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    if rank == 0:
        logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    if args['mode'] == 'train':
        run_train(model, feature_map, params, args)
    elif args['mode'] == 'inference':
        run_inference(model, feature_map, params, args)

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
