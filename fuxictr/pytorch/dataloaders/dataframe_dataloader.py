import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class _PandasDataset(Dataset):
    def __init__(self, feature_map, data):
        self.feature_map = feature_map
        self.df = data[0] if isinstance(data, list) else data
        self.num_samples = len(self.df)
        # Cache columns as numpy arrays to speed up __getitem__
        self.col_arrays = {}
        for col in self.feature_map.features:
            if col in self.df.columns:
                values = self.df[col].values
                if self.df[col].dtype == 'object':
                    values = np.array(self.df[col].to_list())
                self.col_arrays[col] = values

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {col: values[index] for col, values in self.col_arrays.items()}


class DataFrameDataLoader(object):
    def __init__(self, feature_map, data, split="test", batch_size=32, shuffle=False, num_workers=0, **kwargs):
        self.dataset = _PandasDataset(feature_map, data)
        self.num_samples = self.dataset.num_samples
        self.num_blocks = 1
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        # Only pick the worker setting we care about; ignore other kwargs to keep compatibility
        worker_num = kwargs.get('num_workers', num_workers)

        def collate_fn(batch):
            batch_dict = {}
            for col in self.dataset.col_arrays:
                vals = [item[col] for item in batch]
                if isinstance(vals[0], np.ndarray):
                    arr = np.stack(vals)
                else:
                    arr = np.array(vals)
                if np.issubdtype(arr.dtype, np.integer):
                    batch_dict[col] = torch.from_numpy(arr).long()
                elif np.issubdtype(arr.dtype, np.floating):
                    batch_dict[col] = torch.from_numpy(arr).float()
                else:
                    batch_dict[col] = torch.tensor(arr)
            return batch_dict

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle,
                                 num_workers=worker_num, collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.num_batches
