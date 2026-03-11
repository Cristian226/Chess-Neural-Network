import chess
import time
import torch
from threading import Event
from typing import Optional

from ai.encoding import encode_board
from ai.model import ChessEvalNet
from config import *


class NeuralNetEngine:
    def __init__(self, depth, time_limit_ms=AI_SEARCH_TIME_LIMIT_MS):
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cancel_event = Event()
        self._deadline = None
        
        # Load the trained model
        checkpoint = torch.load(AI_MODEL_PATH, map_location=self.device)
        self.model = ChessEvalNet().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.clip_cp = checkpoint.get("clip_cp", 1000.0)

    def evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -99999 if board.turn else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        with torch.no_grad():
            features = encode_board(board).unsqueeze(0).to(self.device)
            pred = self.model(features)
            eval_cp = pred.item() * self.clip_cp # Unscale from [-1, 1] to centipawns
        
        return eval_cp

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        fallback_move = next(iter(board.legal_moves), None)
        self._cancel_event.clear()
        self._deadline = time.perf_counter() + (
            self.time_limit_ms / 1000.0 if self.time_limit_ms else 3600
        )
        maximizing = board.turn == chess.WHITE
        _, move = self.alpha_beta(
            board,
            self.depth,
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=maximizing
        )
        return move or fallback_move

    def evaluate_position(self, board: chess.Board) -> int:
        return int(round(self.evaluate_board(board)))

    def cancel_search(self):
        self._cancel_event.set()

    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        if self._cancel_event.is_set() or time.perf_counter() >= self._deadline:
            return 0, None

        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None

        best_move = None

        if maximizing:
            max_eval = -float("inf")
            for move in board.legal_moves:
                if self._cancel_event.is_set() or time.perf_counter() >= self._deadline:
                    return max_eval, best_move
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in board.legal_moves:
                if self._cancel_event.is_set() or time.perf_counter() >= self._deadline:
                    return min_eval, best_move
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            return min_eval, best_move
