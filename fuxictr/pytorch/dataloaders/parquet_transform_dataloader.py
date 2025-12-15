import glob
import logging
import os

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
from functools import partial


class ParquetTransformIterDataPipe(IterDataPipe):
    """
    Iterates parquet blocks, applies feature_encoder preprocess/transform, and yields row dicts.
    Blocks are partitioned across workers by block index to enable parallel reading.
    """

    def __init__(self, data_blocks, feature_map, feature_encoder, id_cols=None, chunk_size=None):
        self.feature_map = feature_map
        self.feature_encoder = feature_encoder
        self.data_blocks = data_blocks
        self.id_cols = id_cols or []
        self.chunk_size = chunk_size

    def _iter_block_list(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data_blocks
        # Assign blocks to workers by modulo on block index to avoid overlapping work
        return [
            block for idx, block in enumerate(self.data_blocks)
            if idx % worker_info.num_workers == worker_info.id
        ]

    def _prepare_lazyframe(self, path):
        lf = pl.scan_parquet(path, low_memory=False)
        schema_cols = lf.collect_schema().names()

        use_cols = list(self.feature_encoder.dtype_dict.keys())
        if self.feature_encoder.feature_map.group_id is not None and \
           self.feature_encoder.feature_map.group_id not in use_cols:
            use_cols.append(self.feature_encoder.feature_map.group_id)
        # keep only available columns
        use_cols = [c for c in use_cols if c in schema_cols]
        return lf.select(use_cols), [c for c in self.id_cols if c in schema_cols]

    def _yield_rows(self, lf, ids_cols):
        # Collect IDs before transform so order matches
        ids_df = lf.select(ids_cols).collect().to_pandas() if ids_cols else pd.DataFrame()

        df = self.feature_encoder.preprocess(lf).collect().to_pandas()
        df = self.feature_encoder.transform(df)

        n_rows = len(df)
        if n_rows == 0:
            return

        # Optional chunking to limit peak memory when converting to Python dict rows
        step = self.chunk_size or n_rows
        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            df_slice = df.iloc[start:end]
            ids_slice = ids_df.iloc[start:end] if not ids_df.empty else None
            for idx, row in enumerate(df_slice.itertuples(index=False)):
                row_dict = row._asdict()
                if ids_slice is not None:
                    for col in ids_slice.columns:
                        row_dict[col] = ids_slice.iloc[idx][col]
                yield row_dict

    def __iter__(self):
        for block_idx, path in self._iter_block_list():
            lf, ids_cols = self._prepare_lazyframe(path)
            for row_dict in self._yield_rows(lf, ids_cols):
                # add source block index to help downstream grouping
                row_dict["_file_idx"] = block_idx
                yield row_dict


def _collate_batch(batch, feature_cols, meta_cols):
    """Top-level collate to keep picklable for multiprocessing workers."""
    batch_dict = {}
    for key in batch[0].keys():
        vals = [sample[key] for sample in batch]
        if key in feature_cols:
            if isinstance(vals[0], np.ndarray):
                arr = np.stack(vals)
            else:
                arr = np.array(vals)
            if np.issubdtype(arr.dtype, np.integer):
                batch_dict[key] = torch.from_numpy(arr).long()
            elif np.issubdtype(arr.dtype, np.floating):
                batch_dict[key] = torch.from_numpy(arr).float()
            else:
                batch_dict[key] = torch.tensor(arr)
        elif key in meta_cols:
            batch_dict[key] = np.array(vals)
    return batch_dict


class ParquetTransformBlockDataLoader(DataLoader):
    """
    Block-style DataLoader that reads parquet files, applies feature_encoder transforms,
    and streams batches of tensors. Includes id columns and a _file_idx marker so that
    inference can stitch predictions back to source files.
    """

    def __init__(self, feature_map, data_path, split="test", batch_size=32, shuffle=False,
                 num_workers=0, buffer_size=100000, feature_encoder=None, id_cols=None,
                 chunk_size=None, multiprocessing_context=None, **kwargs):
        if feature_encoder is None:
            raise ValueError("feature_encoder is required for ParquetTransformBlockDataLoader")
        if not data_path:
            raise ValueError("data_path is empty.")
        if isinstance(data_path, (list, tuple)):
            data_blocks = list(enumerate(data_path))
        else:
            if not data_path.endswith("parquet"):
                data_path = os.path.join(data_path, "*.parquet")
            data_blocks = list(enumerate(sorted(glob.glob(data_path))))
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"

        # In test/inference we preserve order by default; allow multi-worker if caller insists
        worker_num = kwargs.get("num_workers", num_workers)
        if split == "test" and shuffle:
            logging.warning("Shuffle is disabled in test split for ParquetTransformBlockDataLoader.")
            shuffle = False
        # Build DataPipe
        datapipe = ParquetTransformIterDataPipe(
            data_blocks=data_blocks,
            feature_map=feature_map,
            feature_encoder=feature_encoder,
            id_cols=id_cols,
            chunk_size=chunk_size
        )
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)

        feature_cols = set(feature_map.features.keys())
        meta_cols = set((id_cols or [])) | {"_file_idx"}

        collate_fn = partial(_collate_batch, feature_cols=feature_cols, meta_cols=meta_cols)

        super().__init__(dataset=datapipe, batch_size=batch_size,
                         num_workers=worker_num, collate_fn=collate_fn,
                         multiprocessing_context=multiprocessing_context)
        self.num_blocks = len(data_blocks)
        self.num_samples, self.num_batches = self._count_rows(data_blocks, batch_size)

    def _count_rows(self, data_blocks, batch_size):
        num_rows = 0
        for _, block_path in data_blocks:
            df = pl.scan_parquet(block_path)
            num_rows += df.select(pl.count()).collect().item()
        num_batches = int(np.ceil(num_rows / batch_size))
        return num_rows, num_batches

    def __len__(self):
        return self.num_batches
