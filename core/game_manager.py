import os
import chess
import chess.pgn
import datetime
from typing import Optional

from core.ai_worker import AIMoveWorker, EvalTracker
from engine.engine_selector import get_engine
from config import *


class GameManager:
    def __init__(self):
        self.board = chess.Board()
        self.move_history: list[str] = []
        self.last_move: Optional[chess.Move] = None
        self.mode = GAME_MODE
        self.human_color = HUMAN_COLOR
        self.ai_color = AI_COLOR
        self.white_engine = get_engine(AI_WHITE_ENGINE)
        self.black_engine = get_engine(AI_BLACK_ENGINE)
        self.ai_worker = AIMoveWorker()
        self.sf_eval = EvalTracker(get_engine(STOCKFISH_AI, stockFishElo=STOCKFISH_EVAL_ELO, stockfishTimeLimit=STOCKFISH_EVAL_TIME_LIMIT))
        self.nn_eval = EvalTracker(get_engine(NEURAL_NETWORK_AI))
        self.last_ai_time_ms: Optional[int] = None
        self.pgn_saved = False
        self._schedule_eval()


    def copy_board(self):
        return self.board.copy(stack=True)

    @property
    def move_row_count(self) -> int:
        return (len(self.move_history) + 1) // 2

    def reset(self):
        self.ai_worker.cancel()
        self.board.reset()
        self.move_history.clear()

        self.last_move = None
        self.last_ai_time_ms = None
        self.pgn_saved = False

        self._clear_eval_state()


    def try_move(self, from_sq, to_sq, promotion=None):
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        if move not in self.board.legal_moves:
            return False
        self._push(move)
        return True

    def _push(self, move):
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.move_history.append(san)
        self._schedule_eval()
        self._save_pgn()

    def undo_last_turn(self):
        if self.mode == AI_VS_AI:
            return 0

        self.ai_worker.cancel()
        undo_count = 1 if self.mode == PVP else 2
        undone = 0

        while undone < undo_count and self.board.move_stack:
            self.board.pop()
            undone += 1
            if self.move_history:
                self.move_history.pop()
        self.last_move = self.board.move_stack[-1] if self.board.move_stack else None

        if undone:
            self.last_ai_time_ms = None
            self._clear_eval_state()
        return undone


    def is_ai_turn(self):
        if self.board.is_game_over():
            return False
        if self.mode == PVP:
            return False
        if self.mode == PVE:
            return self.board.turn == self.ai_color
        return True

    def legal_moves_from(self, square):
        return [m for m in self.board.legal_moves if m.from_square == square]

    def set_mode(self, mode):
        if mode not in (PVP, PVE, AI_VS_AI) or mode == self.mode:
            return False

        self.ai_worker.cancel()
        self.ai_worker.shutdown()
        self.ai_worker = AIMoveWorker()
        self.mode = mode
        self.pgn_saved = False
        self.last_ai_time_ms = None

        return True

    def cycle_mode(self):
        modes = (PVP, PVE, AI_VS_AI)
        self.set_mode(modes[(modes.index(self.mode) + 1) % len(modes)])
        return self.mode


    def _current_engine(self):
        return self.white_engine if self.board.turn == chess.WHITE else self.black_engine

    def update_ai_workers(self):
        self.sf_eval.get_eval_from_eval_thread()
        self.nn_eval.get_eval_from_eval_thread()

        if self.ai_worker.get_move_from_engine_thread():
            self.last_ai_time_ms = self.ai_worker.result_time_ms

            if (self.ai_worker.result_fen == self.board.fen()
                and self.is_ai_turn()):

                move = self.ai_worker.result_move
                if move is None:
                    move = next(iter(self.board.legal_moves), None)

                if move:
                    self._push(move)

            self.ai_worker.clear_result()

        board_copy = None
        if not self.sf_eval.busy and self.sf_eval.pending:
            board_copy = board_copy or self.copy_board()
            self.sf_eval.try_request(board_copy)

        if not self.nn_eval.busy and self.nn_eval.pending:
            board_copy = board_copy or self.copy_board()
            self.nn_eval.try_request(board_copy)

        if self.is_ai_turn():
            board_copy = board_copy or self.copy_board()
            self.ai_worker.request(self._current_engine(), board_copy)

    def _clear_eval_state(self):
        self.sf_eval.clear()
        self.nn_eval.clear()
        self._schedule_eval()

    def _schedule_eval(self):
        fen = self.board.fen()
        self.sf_eval.schedule(fen)
        self.nn_eval.schedule(fen)


    def _pgn_player(self, color):
        if self.mode == PVP or (self.mode == PVE and color == self.human_color):
            return "Human"

        return (AI_WHITE_ENGINE if color == chess.WHITE else AI_BLACK_ENGINE).upper()

    def _save_pgn(self):
        if not (self.board.is_game_over() and AI_SAVE_PGN and not self.pgn_saved):
            return

        game = chess.pgn.Game()
        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)

        game.headers["Event"] = self.mode.upper()
        game.headers["Date"] = datetime.date.today().isoformat()
        game.headers["White"] = self._pgn_player(chess.WHITE)
        game.headers["Black"] = self._pgn_player(chess.BLACK)
        game.headers["Result"] = self.board.result()
        game.headers["PlyCount"] = str(len(self.board.move_stack))

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        text = game.accept(exporter)
        os.makedirs(os.path.dirname(AI_PGN_PATH), exist_ok=True)

        with open(AI_PGN_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
        self.pgn_saved = True


    def close(self):
        self.ai_worker.cancel()
        self.ai_worker.shutdown()
        self.sf_eval.shutdown()
        self.nn_eval.shutdown()

        for engine in (self.white_engine, self.black_engine, self.sf_eval.engine, self.nn_eval.engine):
            closer = getattr(engine, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
