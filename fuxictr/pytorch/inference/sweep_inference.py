import logging
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from collections import defaultdict
from tqdm import tqdm


class Inferenceutils:
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _get_preds_dict(return_dict, labels, model=None):
        """Normalize model outputs to a dict of column -> numpy array.

        Args:
            return_dict: Model forward() output dictionary
            labels: Label names from feature_map
            model: Optional model instance for checking task types (binary_classification_logits)

        Note:
            Supports both key formats:
            - f"{label}_pred" (e.g., "click_pred", "conversion_pred")
            - f"{label}" (e.g., "click", "conversion")
        """
        # Single-task model
        if "y_pred" in return_dict:
            pred = return_dict["y_pred"]
            # Apply sigmoid for binary_classification_logits task type
            if model and hasattr(model, 'task_list'):
                if model.task_list[0] == "binary_classification_logits":
                    pred = torch.sigmoid(pred)
            return {"pred": pred.data.cpu().numpy().reshape(-1)}

        # Multi-task model
        preds = {}
        for i, label in enumerate(labels):
            # Try both key formats: "{label}_pred" and "{label}"
            key_pred = f"{label}_pred"
            key_label = label

            # Prefer "{label}_pred" format, fallback to "{label}"
            if key_pred in return_dict:
                pred = return_dict[key_pred]
                output_key = key_pred
            elif key_label in return_dict:
                pred = return_dict[key_label]
                output_key = key_label
            else:
                continue

            # Apply sigmoid for binary_classification_logits task type
            # to ensure output is in (0, 1) range
            if model and hasattr(model, 'task_list') and i < len(model.task_list):
                if model.task_list[i] == "binary_classification_logits":
                    pred = torch.sigmoid(pred)
            preds[output_key] = pred.data.cpu().numpy().reshape(-1)

        # Fallback
        if not preds and "pred" in return_dict:
            preds["pred"] = return_dict["pred"].data.cpu().numpy().reshape(-1)

        if not preds:
            raise KeyError(f"No prediction keys found in model output. Available keys: {list(return_dict.keys())}")
        return preds


class ParquetWriterWrapper:
    def __init__(self, output_dir, buffer_limit=50000, filename_fmt="part_{fid}.parquet"):
        self.output_dir = output_dir
        self.buffer_limit = buffer_limit
        self.filename_fmt = filename_fmt
        self.writers = {}
        self.buffers = defaultdict(list)
        self.buffer_counts = defaultdict(int)

    def write_chunk(self, fid, cache_entry, pred_dict, extra_cols=None):
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
        self.buffers[fid].append(table)
        self.buffer_counts[fid] += row_len
        
        if self.buffer_counts[fid] >= self.buffer_limit:
            self._flush_buffers(fid)

    def _flush_buffers(self, fid):
        if fid not in self.buffers or not self.buffers[fid]:
            return
        tables = self.buffers[fid]
        
        filename = self.filename_fmt.format(fid=int(fid))
        part_path = os.path.join(self.output_dir, filename)
        
        # Open writer only when needed, and append
        if fid not in self.writers:
            # We must use the schema from the first table, assuming consistency
            self.writers[fid] = pq.ParquetWriter(part_path, tables[0].schema)
        
        for tbl in tables:
            self.writers[fid].write_table(tbl)
        
        self.buffers[fid] = []
        self.buffer_counts[fid] = 0

    def close(self):
        # Flush all remaining buffers
        for fid in list(self.buffers.keys()):
            self._flush_buffers(fid)
        # Close all writers
        for w in self.writers.values():
            w.close()
        self.writers.clear()


class SweepInference:
    def __init__(self, model, feature_map, params):
        self.model = model
        self.feature_map = feature_map
        self.params = params
        self.sweep_col = None
        self.valid_indices = []
        self.domains_per_pass = 1
        self.id_to_token = {}
        self._setup_sweep_config()

    def _setup_sweep_config(self):
        sweep_col = self.params.get('domain_feature')
        if not sweep_col and self.params.get('condition_features'):
            sweep_col = self.params['condition_features'][0]
        if not sweep_col:
            sweep_col = 'product'
        
        if sweep_col not in self.feature_map.features:
            logging.warning(f"Sweep column {sweep_col} not in feature_map. Sweep mode disabled.")
            self.sweep_enabled = False
            return
        
        self.sweep_col = sweep_col
        vocab_size = self.feature_map.features[sweep_col]['vocab_size']
        # Typically index 0 is padding/OOV, so range from 1 to vocab_size - 1 or similar
        # Adjusting valid indices logic based on standard assumption
        self.valid_indices = list(range(1, vocab_size)) 
        
        param_domains = self.params.get('sweep_domains_per_pass')
        self.domains_per_pass = max(1, param_domains or len(self.valid_indices))
        self.sweep_enabled = True

    def set_id_to_token(self, feature_encoder):
        if not self.sweep_enabled:
            return
        tokenizer_key = self.sweep_col + "::tokenizer"
        if feature_encoder and tokenizer_key in feature_encoder.processor_dict:
            vocab = feature_encoder.processor_dict[tokenizer_key].vocab
            self.id_to_token = {v: k for k, v in vocab.items()}

    def run_batch(self, batch_data, unique_files, id_cache, writer):
        """
        Runs inference for a single batch, handling sweep logic if enabled.
        """
        # If sweep is disabled, just run normal inference
        if not self.sweep_enabled:
            with torch.no_grad():
                pred_dict = Inferenceutils._get_preds_dict(
                    self.model.forward(batch_data),
                    self.feature_map.labels,
                    model=self.model
                )
            for fid in unique_files:
                writer.write_chunk(fid, id_cache[fid], pred_dict)
            return True

        # Sweep logic
        feature_batch = batch_data 
        if self.sweep_col not in feature_batch:
            logging.warning("Sweep column tensor missing in batch; skipping sweep for this batch.")
            return False

        for start in range(0, len(self.valid_indices), self.domains_per_pass):
            dom_chunk = list(self.valid_indices)[start:start+self.domains_per_pass]
            expanded_batch, base_len = Inferenceutils._repeat_feature_batch(feature_batch, len(dom_chunk))
            
            if base_len == 0:
                continue
                
            sweep_tensor = expanded_batch[self.sweep_col]
            for idx, d_idx in enumerate(dom_chunk):
                start_idx = idx * base_len
                end_idx = start_idx + base_len
                # Modify the sweep column in place
                sweep_tensor[start_idx:end_idx] = d_idx
                
            with torch.no_grad():
                pred_dict = Inferenceutils._get_preds_dict(
                    self.model.forward(expanded_batch),
                    self.feature_map.labels,
                    model=self.model
                )
            
            # Slice back and write
            for idx, d_idx in enumerate(dom_chunk):
                start_idx = idx * base_len
                end_idx = start_idx + base_len
                sliced_pred = {k: v[start_idx:end_idx] for k, v in pred_dict.items()}
                
                dom_label = self.id_to_token.get(d_idx, f"domain_{d_idx}")
                extra = {self.sweep_col: dom_label}
                
                for fid in unique_files:
                    writer.write_chunk(fid, id_cache[fid], sliced_pred, extra_cols=extra)
                    
        return True