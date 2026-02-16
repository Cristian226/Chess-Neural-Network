import csv
from typing import Iterable, Optional, Tuple

import chess
import torch
from torch.utils.data import IterableDataset


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        planes[idx, row, col] = 1.0
    return planes


def parse_eval(eval_str: str) -> Optional[float]:
    eval_str = eval_str.strip()
    if not eval_str:
        return None
    
    # Handle mate notation
    if eval_str.startswith('#'):
        mate_in = int(eval_str[1:])
        return 10000.0 if mate_in > 0 else -10000.0
    
    try:
        return float(eval_str)
    except ValueError:
        return None


class FenDataset(IterableDataset):
    def __init__(self, path: str, max_rows: int):
        super().__init__()
        self.path = path
        self.max_rows = max_rows

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
                raise ValueError("CSV must have 'FEN' and 'Evaluation' columns")

            row_count = 0
            for row in reader:
                if self.max_rows and row_count >= self.max_rows:
                    break
                
                fen = row.get("FEN", "").strip()
                eval_str = row.get("Evaluation", "").strip()
                
                if not fen or not eval_str:
                    print(f"Skipping row with missing FEN or Evaluation: {row}")
                    continue
                
                try:
                    board = chess.Board(fen)
                except ValueError:
                    print(f"Skipping invalid FEN: {fen}")
                    continue
                
                eval_cp = parse_eval(eval_str)
                if eval_cp is None:
                    print(f"Skipping row with invalid Evaluation: {eval_str}")
                    continue
                
                row_count += 1
                features = encode_board(board)
                target = torch.tensor(eval_cp, dtype=torch.float32)
                yield features, target
