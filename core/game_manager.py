import os
import chess
import chess.pgn
import datetime
import json
from typing import Optional
from core.game_state import GameState
from config import *
from engine.engine_selector import get_engine

class GameManager:
    def __init__(self):
        self.game = GameState()
        
        self.white_engine = get_engine(AI_WHITE_ENGINE, AI_WHITE_DEPTH)
        self.black_engine = get_engine(AI_BLACK_ENGINE, AI_BLACK_DEPTH)

        self.next_ai_move_time = 0
        self.pgn_saved = False


    def is_ai_turn(self) -> bool:
        if GAME_MODE == PVP:
            return False
        if GAME_MODE == PVE:
            return self.game.board.turn == AI_COLOR and not self.game.board.is_game_over()
        if GAME_MODE == AI_VS_AI:
            return not self.game.board.is_game_over()

    def is_human_turn(self) -> bool:
        if GAME_MODE == PVP:
            return not self.game.board.is_game_over()
        if GAME_MODE == PVE:
            return self.game.board.turn == HUMAN_COLOR and not self.game.board.is_game_over()
        if GAME_MODE == AI_VS_AI:
            return False

    def reset(self):
        self.game.reset()
        self.next_ai_move_time = 0
        self.pgn_saved = False

    def make_human_move(self, move: chess.Move, now: int) -> bool:
        moved = self.game.make_move(move)
        if moved:
            self.next_ai_move_time = now + AI_MOVE_DELAY_MS
        return moved

    def make_ai_move(self, now: Optional[int] = None):
        if not self.is_ai_turn():
            return

        engine = self.white_engine if self.game.board.turn == chess.WHITE else self.black_engine
        move = engine.get_best_move(self.game.board)
        if move:
            self.game.make_move(move)
            if now is None:
                self.next_ai_move_time = 0
            else:
                self.next_ai_move_time = now + AI_MOVE_DELAY_MS

        if self.game.board.is_game_over() and GAME_MODE == AI_VS_AI and AI_SAVE_PGN and not self.pgn_saved:
            self.save_pgn()
            self.pgn_saved = True

    def update(self, now: int):
        if self.is_ai_turn() and now >= self.next_ai_move_time:
            self.make_ai_move(now)

    def save_pgn(self):
        game = chess.pgn.Game()
        node = game
        for m in self.game.board.move_stack:
            node = node.add_variation(m)

        game.headers["Event"] = GAME_MODE.upper()
        game.headers["Date"] = datetime.date.today().isoformat()
        game.headers["White"] = f"{AI_WHITE_ENGINE} (d={AI_WHITE_DEPTH})"
        game.headers["Black"] = f"{AI_BLACK_ENGINE} (d={AI_BLACK_DEPTH})"
        del game.headers["Site"]
        del game.headers["Round"]

        result = self.game.result() if hasattr(self.game, "result") else self.game.board.result()
        game.headers["Result"] = result
        outcome = self.game.board.outcome()
        if outcome:
            game.headers["Termination"] = outcome.termination.name
            if outcome.winner is not None:
                game.headers["Winner"] = "White" if outcome.winner else "Black"

        game.headers["PlyCount"] = str(len(self.game.board.move_stack))

        try:
            self.stats.finish(result=result)
        except Exception:
            pass

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn_text = game.accept(exporter)

        os.makedirs(os.path.dirname(AI_PGN_PATH), exist_ok=True)
        with open(AI_PGN_PATH, "a", encoding="utf-8") as f:
            f.write(pgn_text)
            f.write("\n\n")

        try:
            stats_line = json.dumps(self.stats.to_dict(), ensure_ascii=False)
            with open(AI_PGN_PATH + ".stats.jsonl", "a", encoding="utf-8") as sf:
                sf.write(stats_line + "\n")
        except Exception:
            pass
