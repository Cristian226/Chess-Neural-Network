import chess
import chess.engine
from typing import Optional
from config import *

class StockfishEngine:
    def __init__(self, elo=STOCKFISH_ELO, time_limit=STOCKFISH_TIME_LIMIT):
        self.engine = None
        self._closed = False
        self.elo = elo
        self.time_limit = time_limit
        try:
            self.start_engine()
        except FileNotFoundError:
            raise RuntimeError("Stockfish not found. Add the path to STOCKFISH_PATH in config.py or add it in your system PATH.")

    def start_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH or "stockfish")
        if self.elo is None:
            self.engine.configure({"UCI_LimitStrength": False})
        else:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.elo})

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        
        result = self.engine.play(board, chess.engine.Limit(time= self.time_limit))
        return result.move

    def evaluate_position(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        info = self.engine.analyse(board, chess.engine.Limit(time=self.time_limit))
        score = info["score"].white().score(mate_score=100000)
        return 0 if score is None else int(score)

    def close(self):
        if self._closed or self.engine is None:
            return

        self._closed = True
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    def cancel_search(self):
        if self.engine is None or self._closed:
            return
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
        self.start_engine()
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
