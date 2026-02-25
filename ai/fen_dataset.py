import csv
from typing import Iterator, Optional, Tuple

import chess
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ai.encoding import encode_board
from ai.training_config import *


def parse_eval(eval_str: str, include_mates: bool, mate_value_cp: float) -> Optional[float]:
    eval_str = eval_str.strip()
    if not eval_str:
        return None

    if eval_str.startswith('#'):
        if not include_mates:
            return None
        try:
            mate_in = int(eval_str[1:])
        except ValueError:
            return None
        return mate_value_cp if mate_in > 0 else -mate_value_cp

    try:
        return float(eval_str)
    except ValueError:
        return None


class FenDataset(IterableDataset):
    def __init__(self, split: str = None):
        super().__init__()
        self.path = DATASET_PATH
        self.max_rows = MAX_ROWS
        self.include_mates = FEN_INCLUDE_MATES
        self.mate_value_cp = FEN_MATE_VALUE_CP
        self.max_abs_eval = FEN_MAX_VALUE_EVAL
        self.split = split
        self.train_cutoff = int(MAX_ROWS * (1.0 - VAL_SPLIT / 100.0))

    def _in_split(self, global_index: int) -> bool:
        if self.split is None:
            return True
        if self.split == "train":
            return global_index < self.train_cutoff
        return global_index >= self.train_cutoff

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
                raise ValueError("CSV must have 'FEN' and 'Evaluation' columns")

            for global_seen, row in enumerate(reader):
                if self.max_rows and global_seen >= self.max_rows:
                    break
                if not self._in_split(global_seen):
                    continue
                if global_seen % num_workers != worker_id:
                    continue

                fen = row.get("FEN", "").strip()
                eval_str = row.get("Evaluation", "").strip()
                if not fen or not eval_str:
                    continue

                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue

                eval_cp = parse_eval(eval_str, self.include_mates, self.mate_value_cp)
                if eval_cp is None:
                    continue
                if self.max_abs_eval is not None and abs(eval_cp) > self.max_abs_eval:
                    continue

                yield encode_board(board), torch.tensor(eval_cp, dtype=torch.float32)
