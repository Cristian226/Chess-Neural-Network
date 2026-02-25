import csv
import io
import re
from typing import Iterator, List, Optional, Tuple

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ai.encoding import encode_board
from ai.training_config import *

EVAL_RE = re.compile(r"\[%eval\s+([+-]?[\d.]+)\]")


def extract_eval(comment: str) -> Optional[float]:
    if not comment:
        return None
    match = EVAL_RE.search(comment)
    if not match:
        return None
    try:
        return float(match.group(1)) * 100.0
    except ValueError:
        return None

def parse_movetext(movetext: str) -> List[Tuple[torch.Tensor, float]]:
    if not movetext:
        return []

    pgn_text = f'[Event "?"]\n\n{movetext}\n'
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return []

    if not game:
        return []

    board = game.board()
    samples = []
    for node in game.mainline():
        board.push(node.move)
        eval_cp = extract_eval(node.comment)
        if eval_cp is not None:
            samples.append((encode_board(board.copy()), eval_cp))
    return samples


class LichessCsvDataset(IterableDataset):
    def __init__(self, split: str = None):
        super().__init__()
        self.path = DATASET_PATH
        self.max_rows = MAX_ROWS
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
            if not reader.fieldnames or "AN" not in reader.fieldnames:
                raise ValueError("CSV missing required 'AN' column")

            for global_seen, row in enumerate(reader):
                if self.max_rows and global_seen >= self.max_rows:
                    break
                if not self._in_split(global_seen):
                    continue
                if global_seen % num_workers != worker_id:
                    continue

                for features, eval_cp in parse_movetext(row.get("AN", "")):
                    yield features, torch.tensor(eval_cp, dtype=torch.float32)
