import chess
import chess.engine
from typing import Optional
from config import *

class StockfishEngine:
    def __init__(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH or "stockfish")
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
        except FileNotFoundError:
            raise RuntimeError("Stockfish not found. Add the path to STOCKFISH_PATH in config.py or add it in your system PATH.")

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        
        result = self.engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
        return result.move
