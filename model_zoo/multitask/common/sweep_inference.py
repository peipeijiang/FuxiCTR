import logging
import pandas as pd
from fuxictr.pytorch.dataloaders import RankDataLoader, DataFrameDataLoader

def sweep_inference(model, feature_map, params, df, sweep_col, feature_encoder=None):
    """
    Perform sweep inference by iterating over all possible values of the sweep column.
    
    Args:
        model: The trained model.
        feature_map: The feature map containing feature specifications.
        params: Configuration parameters.
        df: The input DataFrame (preprocessed).
        sweep_col: The name of the column to sweep (e.g., 'product').
        feature_encoder: Optional FeatureProcessor instance to map IDs back to tokens.
        
    Returns:
        A DataFrame with stacked results and a product column (Long format).
    """
    # Get vocab size
    if sweep_col in feature_map.features:
        # Exclude __PAD__ (0) and __OOV__ (last index)
        # vocab_size includes PAD and OOV, so valid indices are 1 to vocab_size - 2
        vocab_size = feature_map.features[sweep_col]['vocab_size']
        valid_indices = range(1, vocab_size - 1)
    else:
        logging.warning(f"Sweep column {sweep_col} not found in feature map. Skipping sweep.")
        return None
    
    logging.info(f"Starting sweep inference on column: {sweep_col}")
    logging.info(f"Valid indices range: {list(valid_indices)}")

    # Create id_to_token mapping if possible
    id_to_token = {}
    if feature_encoder:
        tokenizer_key = sweep_col + "::tokenizer"
        if tokenizer_key in feature_encoder.processor_dict:
            vocab = feature_encoder.processor_dict[tokenizer_key].vocab
            id_to_token = {v: k for k, v in vocab.items()}
            logging.info(f"Loaded vocabulary for {sweep_col}, size: {len(id_to_token)}")

    # Ensure sweep_col exists in df, if not create it with default value
    if sweep_col not in df.columns:
        df[sweep_col] = 0 # Initialize with 0 (PAD) or any value, will be overwritten

    original_col_data = df[sweep_col].copy()
    sweep_preds = {} # key: domain_idx, value: preds dict
    
    for d_idx in valid_indices:
        # Force set the domain feature
        df[sweep_col] = d_idx 
        
        test_gen = RankDataLoader(feature_map, stage='test', test_data=[df], batch_size=params['batch_size'], 
                            data_loader=DataFrameDataLoader).make_iterator()
        
        model._verbose = 0
        preds = model.predict(test_gen)
        sweep_preds[d_idx] = preds
    
    # Restore original data
    df[sweep_col] = original_col_data
    
    dfs = []
    for d_idx in valid_indices:
        preds = sweep_preds[d_idx]
        # Convert to DataFrame
        batch_df = pd.DataFrame(preds)
        # Add sweep column
        token = id_to_token.get(d_idx, f"domain_{d_idx}")
        batch_df[sweep_col] = token
        dfs.append(batch_df)
    
    if not dfs:
        return None
        
    # Concatenate all products vertically
    return pd.concat(dfs, axis=0, ignore_index=True)
