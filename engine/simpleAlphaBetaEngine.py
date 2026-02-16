import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

class DefaultMinMaxEngine:
    def __init__(self, depth):
        self.depth = depth

    def get_best_move(self, board: chess.Board):
        maximizing = board.turn == chess.WHITE
        _, move = self.alpha_beta(
            board,
            self.depth,
            alpha=-float("inf"),
            beta=float("inf"),
            maximizing=maximizing
        )
        return move

    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None

        best_move = None

        if maximizing:
            max_eval = -float("inf")
            for move in board.legal_moves:
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