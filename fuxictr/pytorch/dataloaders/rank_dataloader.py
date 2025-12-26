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


from .npz_block_dataloader import NpzBlockDataLoader
from .npz_dataloader import NpzDataLoader
from .parquet_block_dataloader import ParquetBlockDataLoader
from .parquet_dataloader import ParquetDataLoader
import logging
import math
from torch.utils.data.distributed import DistributedSampler


class _DistributedDataLoaderWrapper:
    def __init__(self, dataloader, rank, world_size):
        self.dataloader = dataloader
        self.rank = rank % max(world_size, 1)
        self.world_size = max(world_size, 1)
        self._mirror_attributes()

    def _mirror_attributes(self):
        for attr in ["num_samples", "num_blocks", "num_batches", "batch_size"]:
            if hasattr(self.dataloader, attr):
                value = getattr(self.dataloader, attr)
                if isinstance(value, int) and value > 0 and attr != "batch_size":
                    setattr(self, attr, max(1, math.ceil(value / self.world_size)))
                else:
                    setattr(self, attr, value)

    def __iter__(self):
        for idx, batch in enumerate(self.dataloader):
            if idx % self.world_size == self.rank:
                yield batch

    def __len__(self):
        if hasattr(self.dataloader, "__len__"):
            base_len = len(self.dataloader)
            return max(1, math.ceil(base_len / self.world_size)) if base_len else 0
        return 0

    def __getattr__(self, item):
        return getattr(self.dataloader, item)


class RankDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, streaming=False, data_format="npz", **kwargs):
        self._distributed_rank = kwargs.pop("distributed_rank", 0)
        self._distributed_world_size = kwargs.pop("distributed_world_size", 1)
        self._distributed = self._distributed_world_size > 1
        if self._distributed_rank == 0:
            logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        if kwargs.get("data_loader"):
            DataLoader = kwargs["data_loader"]
        else:
            if data_format == "npz":
                DataLoader = NpzBlockDataLoader if streaming else NpzDataLoader
            else: # ["parquet", "csv"]
                DataLoader = ParquetBlockDataLoader if streaming else ParquetDataLoader
        self.stage = stage
        train_sampler = None
        # 若未显式传入 drop_last，分布式默认开启以保证各 rank 步数对齐；单机默认关闭
        train_drop_last = kwargs.pop("drop_last", None)
        if train_drop_last is None:
            train_drop_last = True if self._distributed else False
        train_sampler_kwargs = None
        eval_sampler_kwargs = None
        if self._distributed and DataLoader in (ParquetDataLoader, NpzDataLoader):
            train_sampler_kwargs = {
                "num_replicas": self._distributed_world_size,
                "rank": self._distributed_rank,
                "shuffle": shuffle,
                "drop_last": train_drop_last,
            }
            eval_sampler_kwargs = {
                "num_replicas": self._distributed_world_size,
                "rank": self._distributed_rank,
                "shuffle": False,
                "drop_last": False,
            }
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, split="train", batch_size=batch_size,
                                   shuffle=False if self._distributed else shuffle,
                                   sampler=train_sampler,
                                   drop_last=train_drop_last,
                                   sampler_kwargs=train_sampler_kwargs,
                                   **kwargs)
            if self._distributed_rank == 0:
                logging.info(
                    "Train samples: total/{:d}, blocks/{:d}"
                    .format(train_gen.num_samples, train_gen.num_blocks)
                )     
            if valid_data:
                valid_gen = DataLoader(feature_map, valid_data, split="valid",
                                       batch_size=batch_size, shuffle=False, drop_last=False,
                                       sampler=None,
                                       sampler_kwargs=eval_sampler_kwargs,
                                       **kwargs)
                if self._distributed_rank == 0:
                    logging.info(
                        "Validation samples: total/{:d}, blocks/{:d}"
                        .format(valid_gen.num_samples, valid_gen.num_blocks)
                    )

        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, split="test", batch_size=batch_size,
                                      shuffle=False, drop_last=False,
                                      sampler=None,
                                      sampler_kwargs=eval_sampler_kwargs,
                                      **kwargs)
                if self._distributed_rank == 0:
                    logging.info(
                        "Test samples: total/{:d}, blocks/{:d}"
                        .format(test_gen.num_samples, test_gen.num_blocks)
                    )
        self.train_gen = self._wrap_distributed(train_gen)
        self.valid_gen = self._wrap_distributed(valid_gen)
        self.test_gen = self._wrap_distributed(test_gen)

    def make_iterator(self):
        if self.stage == "train":
            if self._distributed_rank == 0:
                logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            if self._distributed_rank == 0:
                logging.info("Loading test data done.")
            return self.test_gen
        else:
            if self._distributed_rank == 0:
                logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen

    def _wrap_distributed(self, generator):
        if not self._distributed or generator is None:
            return generator
        # 使用 DistributedSampler 时无需再 wrap（当前 DataLoader 暂未集成 sampler，保留兼容）
        if hasattr(generator, "sampler") and isinstance(generator.sampler, DistributedSampler):
            return generator
        return _DistributedDataLoaderWrapper(generator, self._distributed_rank, self._distributed_world_size)
