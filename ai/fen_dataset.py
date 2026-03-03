import csv
from typing import Iterator, Optional, Tuple

import chess
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ai.encoding import encode_board
from ai.training_config import *


def parse_eval(eval_str: str) -> Optional[float]:
    eval_str = eval_str.strip()
    if not eval_str:
        return None

    if eval_str.startswith('#'):
        if not FEN_INCLUDE_MATES:
            return None
        try:
            mate_in = int(eval_str[1:])
        except ValueError:
            return None
        return FEN_MATE_VALUE_CP if mate_in > 0 else -FEN_MATE_VALUE_CP

    try:
        return float(eval_str)
    except ValueError:
        return None


class FenDataset(IterableDataset):
    def __init__(self, split: str = None):
        super().__init__()
        self.split = split
        self.train_cutoff = int(MAX_ROWS_FEN * (1.0 - VAL_SPLIT / 100.0))

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

        with open(FEN_PATH, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
                raise ValueError("CSV must have 'FEN' and 'Evaluation' columns")

            for global_seen, row in enumerate(reader):
                if MAX_ROWS_FEN and global_seen >= MAX_ROWS_FEN:
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

                eval_cp = parse_eval(eval_str)
                if eval_cp is None:
                    continue
                if FEN_MAX_VALUE_EVAL is not None and abs(eval_cp) > FEN_MAX_VALUE_EVAL:
                    continue

                yield encode_board(board), torch.tensor(eval_cp, dtype=torch.float32)
