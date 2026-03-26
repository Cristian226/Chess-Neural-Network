import os
from glob import glob
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedShardDataset(IterableDataset):
    def __init__(self, split: str, data_dir: str) -> None:
        super().__init__()
        self.split = split
        self.data_dir = data_dir

        pattern = os.path.join(data_dir, f"{split}_*.pt")
        self.shards = sorted(glob(pattern))
        if not self.shards:
            raise FileNotFoundError(f"No shards found for split '{split}' in '{data_dir}'.")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        for idx, shard_path in enumerate(self.shards):
            if idx % num_workers != worker_id:
                continue
            payload = torch.load(shard_path, map_location="cpu")
            features, targets = payload["features"], payload["targets"]
            yield from zip(features, targets)
