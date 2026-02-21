import csv
import io
import re
from typing import Iterable, List, Optional, Tuple

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
    
    pgn_text = f"[Event \"?\"]\n\n{movetext}\n"
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except:
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
    def __init__(self):
        super().__init__()
        self.path = DATASET_PATH
        self.max_rows = MAX_ROWS
        self.min_elo = MIN_ELO

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "AN" not in reader.fieldnames:
                raise ValueError("CSV missing required 'AN' column")

            global_seen = 0
            for row in reader:
                if self.max_rows and global_seen >= self.max_rows:
                    break

                if global_seen % num_workers != worker_id:
                    global_seen += 1
                    continue
                
                if self.min_elo:
                    try:
                        if int(row.get("WhiteElo", 0)) < self.min_elo or int(row.get("BlackElo", 0)) < self.min_elo:
                            global_seen += 1
                            continue
                    except ValueError:
                        global_seen += 1
                        continue
                
                global_seen += 1
                for features, eval_cp in parse_movetext(row.get("AN", "")):
                    yield features, torch.tensor(eval_cp, dtype=torch.float32)