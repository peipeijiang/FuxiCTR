import logging
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from collections import defaultdict
from fuxictr.pytorch.dataloaders import RankDataLoader, ParquetTransformBlockDataLoader
from tqdm import tqdm


def _prepare_id_cache(ids_batch, unique_files, file_indices):
    """Cache id columns and masks per file to avoid recomputation."""
    cache = {}
    for fid in unique_files:
        mask = file_indices == fid
        id_dict = {}
        if ids_batch:
            for key, values in ids_batch.items():
                id_dict[key] = np.array(values)[mask]
        cache[fid] = {
            "ids": id_dict,
            "mask": mask,
            "length": int(mask.sum())
        }
    return cache


def _repeat_feature_batch(feature_batch, repeat_factor):
    """Repeat every feature along batch dimension repeat_factor times."""
    expanded = {}
    base_length = None
    for key, value in feature_batch.items():
        if torch.is_tensor(value):
            expanded_tensor = value.repeat_interleave(repeat_factor, dim=0)
            expanded[key] = expanded_tensor
            if base_length is None:
                base_length = value.shape[0]
        else:
            arr = np.asarray(value)
            expanded_arr = np.repeat(arr, repeat_factor, axis=0)
            expanded[key] = expanded_arr
            if base_length is None:
                base_length = len(arr)
    return expanded, base_length or 0


def _write_predictions(fid, output_dir, cache_entry, pred_dict, buffers, buffer_counts, buffer_limit, writers, extra_cols=None):
    """Append prediction chunk for a specific file using cached ids and masks."""
    payload = {}
    ids_dict = cache_entry.get("ids", {})
    if ids_dict:
        payload.update(ids_dict)

    mask = cache_entry["mask"]
    row_len = cache_entry["length"]
    for col, arr in pred_dict.items():
        payload[col] = arr[mask]

    if extra_cols:
        for col, value in extra_cols.items():
            if isinstance(value, np.ndarray) and len(value) == row_len:
                payload[col] = value
            else:
                payload[col] = np.repeat(value, row_len)

    table = pa.Table.from_pydict(payload)
    buffers[fid].append(table)
    buffer_counts[fid] += row_len
    if buffer_counts[fid] >= buffer_limit:
        _flush_buffers(fid, output_dir, writers, buffers, buffer_counts)


def _flush_buffers(fid, output_dir, writers, buffers, buffer_counts):
    if fid not in buffers or not buffers[fid]:
        return
    tables = buffers[fid]
    part_path = os.path.join(output_dir, f"part_{int(fid)}.parquet")
    if fid not in writers:
        writers[fid] = pq.ParquetWriter(part_path, tables[0].schema)
    for tbl in tables:
        writers[fid].write_table(tbl)
    buffers[fid] = []
    buffer_counts[fid] = 0


def _streaming_parquet_inference(model, feature_map, feature_encoder, params, files, output_dir, id_cols, sweep=False):
    """
    Stream parquet blocks, apply preprocess/transform inside DataLoader, and predict.
    Supports normal and sweep inference.
    """
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        test_gen = RankDataLoader(
            feature_map,
            stage='test',
            test_data=files,
            batch_size=params['batch_size'],
            num_workers=params.get('num_workers', 0),
            multiprocessing_context="spawn",
            data_loader=ParquetTransformBlockDataLoader,
            feature_encoder=feature_encoder,
            id_cols=id_cols,
            chunk_size=params.get('infer_chunk_size')
        ).make_iterator()

        writers = {}
        buffers = defaultdict(list)
        buffer_counts = defaultdict(int)
        buffer_limit = params.get('sweep_write_buffer_rows', 50000)
        model._verbose = 0
        has_data = False

        # Prepare sweep domain info if needed
        if sweep:
            sweep_col = params.get('domain_feature')
            if not sweep_col and params.get('condition_features'):
                sweep_col = params['condition_features'][0]
            if not sweep_col:
                sweep_col = 'product'
            if sweep_col not in feature_map.features:
                logging.warning(f"Sweep column {sweep_col} not in feature_map. Skip sweep.")
                sweep = False
            else:
                vocab_size = feature_map.features[sweep_col]['vocab_size']
                valid_indices = list(range(1, vocab_size - 1))
                param_domains = params.get('sweep_domains_per_pass')
                domains_per_pass = max(1, param_domains or len(valid_indices))
                id_to_token = {}
                tokenizer_key = sweep_col + "::tokenizer"
                if feature_encoder and tokenizer_key in feature_encoder.processor_dict:
                    vocab = feature_encoder.processor_dict[tokenizer_key].vocab
                    id_to_token = {v: k for k, v in vocab.items()}

        iterator = test_gen
        if params.get('verbose', 1) > 0:
            iterator = tqdm(test_gen, desc="Streaming inference", mininterval=1.0)

        for batch_data in iterator:
            file_idx_arr = batch_data.pop("_file_idx")
            ids_batch = {c: batch_data.pop(c) for c in list(batch_data.keys()) if c in id_cols}
            feature_batch = {k: v for k, v in batch_data.items() if k in feature_map.features}

            file_indices = np.array(file_idx_arr)
            unique_files = np.unique(file_indices)
            id_cache = _prepare_id_cache(ids_batch, unique_files, file_indices)

            if sweep:
                if sweep_col not in feature_batch:
                    logging.warning("Sweep column tensor missing in batch; skipping sweep for this batch.")
                    continue
                for start in range(0, len(valid_indices), domains_per_pass):
                    dom_chunk = list(valid_indices)[start:start+domains_per_pass]
                    expanded_batch, base_len = _repeat_feature_batch(feature_batch, len(dom_chunk))
                    if base_len == 0:
                        continue
                    sweep_tensor = expanded_batch[sweep_col]
                    for idx, d_idx in enumerate(dom_chunk):
                        start_idx = idx * base_len
                        end_idx = start_idx + base_len
                        sweep_tensor[start_idx:end_idx] = d_idx
                    with torch.no_grad():
                        pred_dict = _get_preds_dict(model.forward(expanded_batch), feature_map.labels)
                    for idx, d_idx in enumerate(dom_chunk):
                        start_idx = idx * base_len
                        end_idx = start_idx + base_len
                        sliced_pred = {k: v[start_idx:end_idx] for k, v in pred_dict.items()}
                        dom_label = id_to_token.get(d_idx, f"domain_{d_idx}")
                        for fid in unique_files:
                            extra = {sweep_col: dom_label}
                            _write_predictions(
                                fid,
                                output_dir,
                                id_cache[fid],
                                sliced_pred,
                                buffers,
                                buffer_counts,
                                buffer_limit,
                                writers,
                                extra_cols=extra
                            )
                has_data = True
            else:
                with torch.no_grad():
                    pred_dict = _get_preds_dict(model.forward(feature_batch), feature_map.labels)
                has_data = True

                for fid in unique_files:
                    _write_predictions(
                        fid,
                        output_dir,
                        id_cache[fid],
                        pred_dict,
                        buffers,
                        buffer_counts,
                        buffer_limit,
                        writers
                    )

        # Flush any buffered tables before closing writers
        for fid in list(buffers.keys()):
            _flush_buffers(fid, output_dir, writers, buffers, buffer_counts)
        for w in writers.values():
            w.close()
        model._verbose = params.get('verbose', 1)
        return has_data
    finally:
        logger.setLevel(original_level)


def run_inference_files(model, feature_map, feature_encoder, params, args, files, output_dir):
    """
    Unified inference entry for parquet inputs.
    Always uses block streaming with optional sweep; other formats are not supported here.
    """
    data_format = 'parquet' if files[0].endswith('.parquet') else 'csv'
    id_cols = [c for c in ['phone', 'phone_md5'] if c in feature_encoder.dtype_dict]
    if data_format != 'parquet':
        raise ValueError("Only parquet inference is supported in streaming mode.")

    return _streaming_parquet_inference(
        model, feature_map, feature_encoder, params,
        files, output_dir, id_cols, sweep=args.get('sweep', False))


def _get_preds_dict(return_dict, labels):
    """Normalize model outputs to a dict of column -> numpy array."""
    if "y_pred" in return_dict:
        return {"pred": return_dict["y_pred"].data.cpu().numpy().reshape(-1)}
    preds = {}
    for label in labels:
        key = f"{label}_pred"
        if key in return_dict:
            preds[key] = return_dict[key].data.cpu().numpy().reshape(-1)
    if not preds:
        raise KeyError("No prediction keys found in model output.")
    return preds
