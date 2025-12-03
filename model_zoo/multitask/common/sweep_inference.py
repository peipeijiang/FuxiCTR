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
    
    # Optimization: Process domains in chunks to vectorize operations
    # Heuristic: We want total rows in a "super-batch" to be reasonable to avoid OOM.
    # Assuming ~1M rows is safe for typical memory.
    MAX_ROWS_PER_PASS = 1000000
    n_rows = len(df)
    if n_rows == 0: return None
    
    domains_per_pass = max(1, MAX_ROWS_PER_PASS // n_rows)
    logging.info(f"Optimized sweep: Processing {domains_per_pass} domains per pass (Input rows: {n_rows})")

    all_results = []
    valid_indices_list = list(valid_indices)
    
    import numpy as np
    
    for i in range(0, len(valid_indices_list), domains_per_pass):
        chunk_indices = valid_indices_list[i : i + domains_per_pass]
        
        # Create a super-batch by replicating the dataframe
        # We use list comprehension + concat which is generally efficient
        chunk_dfs = []
        for d_idx in chunk_indices:
            # We can optimize this further by not copying the whole DF if we could 
            # construct the batch tensor directly, but sticking to DataFrame interface for compatibility
            # Using assign is cleaner but copy() + assignment is robust
            temp_df = df.copy()
            temp_df[sweep_col] = d_idx
            chunk_dfs.append(temp_df)
        
        super_batch_df = pd.concat(chunk_dfs, axis=0, ignore_index=True)
        
        # Run inference on the super-batch
        # RankDataLoader will handle batching this super-batch into GPU-sized chunks
        test_gen = RankDataLoader(feature_map, stage='test', test_data=[super_batch_df], 
                                batch_size=params['batch_size'], 
                                data_loader=DataFrameDataLoader).make_iterator()
        
        model._verbose = 0
        preds = model.predict(test_gen)
        
        # preds is a dict or array. We need to split it back to assign to domains
        # But actually, we can just construct the result dataframe directly from preds
        # because the order is preserved (domain 1 rows, domain 2 rows, ...)
        
        if isinstance(preds, dict):
            batch_res_df = pd.DataFrame(preds)
        else:
            batch_res_df = pd.DataFrame(preds, columns=['pred']) # fallback name
            
        # Add the sweep column to the result
        # We need to repeat the domain tokens matching the rows
        domain_tokens = []
        for d_idx in chunk_indices:
            token = id_to_token.get(d_idx, f"domain_{d_idx}")
            domain_tokens.extend([token] * n_rows)
            
        batch_res_df[sweep_col] = domain_tokens
        all_results.append(batch_res_df)
        
        # Explicit GC
        del super_batch_df, chunk_dfs, test_gen, preds
        import gc
        gc.collect()
    
    # Restore original data
    df[sweep_col] = original_col_data
    
    if not all_results:
        return None
        
    # Concatenate all results vertically
    return pd.concat(all_results, axis=0, ignore_index=True)
