import chess
import time
from threading import Event
from config import AI_SEARCH_TIME_LIMIT_MS

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

class DefaultMinMaxEngine:
    def __init__(self, depth, time_limit_ms=AI_SEARCH_TIME_LIMIT_MS):
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self._cancel_event = Event()
        self._deadline = None

    def get_best_move(self, board: chess.Board):
        fallback_move = next(iter(board.legal_moves), None)
        self._cancel_event.clear()
        self._deadline = time.perf_counter() + (self.time_limit_ms / 1000.0 if self.time_limit_ms else 3600)
        maximizing = board.turn == chess.WHITE
        _, move = self.alpha_beta(
            board,
            self.depth,
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=maximizing
        )
        return move or fallback_move

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
                eval_score, _ = self.alpha_beta(board, depth-1, alpha, beta, False)
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
                eval_score, _ = self.alpha_beta(board, depth-1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def evaluate_board(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -99999 if board.turn else 99999

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value

        return score

    def evaluate_position(self, board: chess.Board) -> int:
        return self.evaluate_board(board)

    def cancel_search(self):
        self._cancel_event.set()