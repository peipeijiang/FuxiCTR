import torch
import numpy as np

class DataFrameDataLoader(object):
    def __init__(self, feature_map, data, split="test", batch_size=32, shuffle=False, **kwargs):
        self.feature_map = feature_map
        self.df = data[0] if isinstance(data, list) else data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.df)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self.num_samples)
            batch_df = self.df.iloc[start:end]
            batch_dict = {}
            for col in self.feature_map.features:
                if col in batch_df.columns:
                    val = batch_df[col].values
                    if batch_df[col].dtype == 'object':
                        val = np.array(batch_df[col].to_list())
                    
                    if np.issubdtype(val.dtype, np.integer):
                        batch_dict[col] = torch.from_numpy(val).long()
                    elif np.issubdtype(val.dtype, np.floating):
                        batch_dict[col] = torch.from_numpy(val).float()
                    else:
                        batch_dict[col] = torch.tensor(val)
            yield batch_dict

    def __len__(self):
        return self.num_batches
