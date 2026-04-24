import os
from glob import glob
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ai.training_config import MAX_PREPROCESS_POSITIONS, VAL_SPLIT


class PreprocessedShardDataset(IterableDataset):
    def __init__(self, split: str, data_dirs: list[str]):
        super().__init__()
        self.split = split
        self.data_dirs = data_dirs
        self.shards = []
        self.max_positions = None

        if MAX_PREPROCESS_POSITIONS:
            train_limit = int(MAX_PREPROCESS_POSITIONS * (1.0 - VAL_SPLIT / 100.0))
            if split == "train":
                self.max_positions = train_limit
            elif split == "val":
                self.max_positions = MAX_PREPROCESS_POSITIONS - train_limit
            else:
                self.max_positions = MAX_PREPROCESS_POSITIONS

        for directory in self.data_dirs:
            pattern = os.path.join(directory, f"{split}_*.pt")
            self.shards.extend(glob(pattern))

        if not self.shards:
            raise FileNotFoundError(f"No shards found for split '{split}'")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        global_seen = 0
        for shard_path in self.shards:
            payload = torch.load(shard_path, map_location="cpu")
            features, targets = payload["features"], payload["targets"]

            for feature, target in zip(features, targets):
                if self.max_positions and global_seen >= self.max_positions:
                    return
                if global_seen % num_workers != worker_id:
                    global_seen += 1
                    continue

                yield feature, target
                global_seen += 1
