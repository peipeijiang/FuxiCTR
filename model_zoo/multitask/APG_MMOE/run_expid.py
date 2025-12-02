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
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sweep_inference import sweep_inference # Import the new module

def run_train(model, feature_map, params, args):
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    logging.info('******** Test evaluation ********')
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
      test_result = model.evaluate(test_gen)
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), args['expid'], params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))

def run_inference(model, feature_map, params, args):
    model.load_weights(model.checkpoint)
    
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_encoder = FeatureProcessor(**params).load_pickle(params.get('pickle_file', os.path.join(data_dir, "feature_processor.pkl")))
    feature_encoder.dtype_dict.update({'phone': str, 'phone_md5': str})

    infer_data = params['infer_data']
    if os.path.isdir(infer_data):
        files = sorted(glob.glob(os.path.join(infer_data, "*.parquet"))) or sorted(glob.glob(os.path.join(infer_data, "*.csv")))
    else:
        files = [infer_data]
    data_format = 'parquet' if files[0].endswith('.parquet') else 'csv'

    # Output directory setup (Parquet format for Big Data)
    output_dir = os.path.join(data_dir, f"{args['expid']}_inference_result")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('******** Start Inference ********')
    logging.info(f"Results will be saved to directory: {output_dir}")
    
    import warnings
    from tqdm import tqdm
    warnings.simplefilter("ignore")
    logger = logging.getLogger()
    original_level = logger.level

    has_data = False

    for i, f in enumerate(tqdm(files, desc="Inference")):
        logger.setLevel(logging.WARNING)
        try:
            ddf = feature_encoder.read_data(f, data_format=data_format)
            
            # Extract IDs before preprocess (which filters columns)
            ids = ddf.select([c for c in ['phone', 'phone_md5'] if c in ddf.columns]).collect().to_pandas()
            
            # Preprocess (handles sequence conversion) and Transform
            df = feature_encoder.preprocess(ddf).collect().to_pandas()
            df = feature_encoder.transform(df)
            
            current_batch_preds = None
            if args.get('sweep', False):
                # Identify sweep feature
                sweep_col = params.get('domain_feature')
                if not sweep_col and params.get('condition_features'):
                    sweep_col = params['condition_features'][0]
                if not sweep_col:
                    sweep_col = 'product' # default
                
                current_batch_preds = sweep_inference(model, feature_map, params, df, sweep_col, feature_encoder=feature_encoder)
            else:
                test_gen = RankDataLoader(feature_map, stage='test', test_data=[df], batch_size=params['batch_size'], 
                                        data_loader=DataFrameDataLoader).make_iterator()
                
                model._verbose = 0
                current_batch_preds = model.predict(test_gen)
            
            if current_batch_preds is not None:
                has_data = True
                
                if isinstance(current_batch_preds, pd.DataFrame):
                    # Long format sweep result
                    pred_df = current_batch_preds
                    
                    # Expand IDs to match the number of rows in pred_df
                    n_rows_pred = len(pred_df)
                    n_rows_ids = len(ids)
                    if n_rows_ids > 0:
                        n_repeats = n_rows_pred // n_rows_ids
                        ids_expanded = pd.concat([ids] * n_repeats, axis=0, ignore_index=True)
                    else:
                        ids_expanded = pd.DataFrame()
                        
                    result_df = pd.concat([ids_expanded, pred_df], axis=1)
                else:
                    # Dict (Normal inference)
                    pred_df = pd.DataFrame(current_batch_preds)
                    result_df = pd.concat([ids.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

                # Save as Parquet part file
                part_file = os.path.join(output_dir, f"part_{i}.parquet")
                result_df.to_parquet(part_file, index=False)
            
            model._verbose = params.get('verbose', 1)
            
            # Explicit garbage collection
            del ddf, df, ids, current_batch_preds
            gc.collect()

        finally:
            logger.setLevel(original_level)

    if has_data:
        logging.info(f"Inference completed. Data saved in: {output_dir}")
    else:
        logging.warning("No data found in infer_data!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='APG_MMOE_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='The running mode.')
    parser.add_argument('--sweep', action='store_true', help='Whether to sweep the domain feature for inference.')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "parquet" and args['mode'] == 'train':
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    if args['mode'] == 'train':
        run_train(model, feature_map, params, args)
    elif args['mode'] == 'inference':
        run_inference(model, feature_map, params, args)
