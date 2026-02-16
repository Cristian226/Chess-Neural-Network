import csv
import io
import re
from typing import Iterable, List, Optional, Tuple

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset

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


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        planes[idx, row, col] = 1.0
    return planes


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
    
    def __init__(self, path: str, max_rows: Optional[int] = None, min_elo: Optional[int] = None):
        super().__init__()
        self.path = path
        self.max_rows = max_rows
        self.min_elo = min_elo

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "AN" not in reader.fieldnames:
                raise ValueError("CSV missing required 'AN' column")

            row_count = 0
            for row in reader:
                if self.max_rows and row_count >= self.max_rows:
                    break
                
                if self.min_elo:
                    try:
                        if int(row.get("WhiteElo", 0)) < self.min_elo or int(row.get("BlackElo", 0)) < self.min_elo:
                            continue
                    except ValueError:
                        continue
                
                row_count += 1
                for features, eval_cp in parse_movetext(row.get("AN", "")):
                    yield features, torch.tensor(eval_cp, dtype=torch.float32)