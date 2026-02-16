import chess
import chess.engine
from typing import Optional
from config import *

class StockfishEngine:
    def __init__(self):
        self.depth = STOCKFISH_DEPTH
        self.time_limit = STOCKFISH_TIME_LIMIT
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH or "stockfish")
        except FileNotFoundError:
            raise RuntimeError("Stockfish not found. Add the path to STOCKFISH_PATH in config.py or add it in your system PATH.")

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth, time=self.time_limit))
        return result.move
