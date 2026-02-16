import chess

class GameState:
    def __init__(self):
        self.board = chess.Board()
        self.last_move = None

    def reset(self):
        self.board.reset()

    def make_move(self, move: chess.Move):
        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            return True
        return False

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        return self.board.result()

