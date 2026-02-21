import csv
from typing import Iterable, Optional, Tuple

import chess
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ai.encoding import encode_board
from ai.training_config import *

def parse_eval(eval_str: str, include_mates: bool, mate_value_cp: float) -> Optional[float]:
    eval_str = eval_str.strip()
    if not eval_str:
        return None
    
    # Handle mate notation
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
    def __init__(self):
        super().__init__()
        self.path = DATASET_PATH
        self.max_rows = MAX_ROWS
        self.include_mates = FEN_INCLUDE_MATES
        self.mate_value_cp = FEN_MATE_VALUE_CP
        self.max_abs_eval = FEN_MAX_VALUE_EVAL

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
                raise ValueError("CSV must have 'FEN' and 'Evaluation' columns")

            global_seen = 0
            for row in reader:
                if self.max_rows and global_seen >= self.max_rows:
                    break

                if global_seen % num_workers != worker_id:
                    global_seen += 1
                    continue
                
                fen = row.get("FEN", "").strip()
                eval_str = row.get("Evaluation", "").strip()
                
                if not fen or not eval_str:
                    global_seen += 1
                    continue
                
                try:
                    board = chess.Board(fen)
                except ValueError:
                    global_seen += 1
                    continue
                
                eval_cp = parse_eval(eval_str, self.include_mates, self.mate_value_cp)
                if eval_cp is None:
                    global_seen += 1
                    continue

                if self.max_abs_eval is not None and abs(eval_cp) > self.max_abs_eval:
                    global_seen += 1
                    continue
                
                global_seen += 1
                features = encode_board(board)
                target = torch.tensor(eval_cp, dtype=torch.float32)
                yield features, target
