import os
import chess
import chess.pgn
import datetime
from typing import List, Optional

from core.ai_worker import AIMoveWorker, EvalTracker
from config import *
from engine.engine_selector import get_engine


class GameManager:
    def __init__(self):
        self.board = chess.Board()
        self.last_move: Optional[chess.Move] = None
        self.move_history: List[str] = []
        self.mode = GAME_MODE
        self.human_color = HUMAN_COLOR
        self.ai_color = AI_COLOR
        self.white_engine = get_engine(AI_WHITE_ENGINE)
        self.black_engine = get_engine(AI_BLACK_ENGINE)
        self.ai_worker = AIMoveWorker()
        self.sf_eval = EvalTracker(get_engine(STOCKFISH_AI, stockFishElo=STOCKFISH_EVAL_ELO,
                stockfishTimeLimit=STOCKFISH_EVAL_TIME_LIMIT))
        self.nn_eval = EvalTracker(get_engine(NEURAL_NETWORK_AI))
        self.pgn_saved = False
        self.last_ai_time_ms: Optional[int] = None
        self._schedule_evaluation()

    def _schedule_evaluation(self):
        fen = self.board.fen()
        self.sf_eval.schedule(fen)
        self.nn_eval.schedule(fen)

    def _request_board(self, board_copy: Optional[chess.Board]) -> chess.Board:
        return board_copy if board_copy is not None else self.copy_board()

    def _clear_state(self):
        self.pgn_saved = False
        self.last_ai_time_ms = None
        self.sf_eval.clear()
        self.nn_eval.clear()

    def _apply_position_change(self):
        self._schedule_evaluation()
        self._save_pgn()

    def copy_board(self) -> chess.Board:
        return self.board.copy(stack=True)

    def legal_moves_from(self, square: int) -> List[chess.Move]:
        return [move for move in self.board.legal_moves if move.from_square == square]

    def result(self) -> str:
        return self.board.result()

    @property
    def move_row_count(self) -> int:
        return (len(self.move_history) + 1) // 2

    def _push_move(self, move: chess.Move) -> bool:
        if move not in self.board.legal_moves:
            return False
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.move_history.append(san)
        return True

    def _undo_moves(self, count: int) -> int:
        undone = 0
        while undone < count and self.board.move_stack:
            self.board.pop()
            undone += 1
            if self.move_history:
                self.move_history.pop()
        self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
        return undone

    def _current_engine(self):
        return self.white_engine if self.board.turn == chess.WHITE else self.black_engine

    def _fallback_legal_move(self) -> Optional[chess.Move]:
        return next(iter(self.board.legal_moves), None)

    def _pgn_player_name(self, color: bool) -> str:
        if self.mode == PVP or (self.mode == PVE and color == self.human_color):
            return "Human"
        return (AI_WHITE_ENGINE if color == chess.WHITE else AI_BLACK_ENGINE).upper()

    def _save_pgn(self):
        if self.board.is_game_over() and self.mode == AI_VS_AI and AI_SAVE_PGN and not self.pgn_saved:
            self.save_pgn()
            self.pgn_saved = True

    def is_ai_turn(self) -> bool:
        if self.board.is_game_over():
            return False
        if self.mode == PVP:
            return False
        if self.mode == PVE:
            return self.board.turn == self.ai_color
        return self.mode == AI_VS_AI

    def is_human_turn(self) -> bool:
        return not self.board.is_game_over() and not self.is_ai_turn()

    def set_mode(self, mode: str) -> bool:
        if mode not in (PVP, PVE, AI_VS_AI) or mode == self.mode:
            return False
        self.ai_worker.cancel()
        self.ai_worker.shutdown()
        self.ai_worker = AIMoveWorker()
        self.mode = mode
        self.pgn_saved = False
        self.last_ai_time_ms = None
        return True

    def cycle_mode(self) -> str:
        modes = (PVP, PVE, AI_VS_AI)
        self.set_mode(modes[(modes.index(self.mode) + 1) % len(modes)])
        return self.mode

    def reset(self):
        self.ai_worker.cancel()
        self.board.reset()
        self.last_move = None
        self.move_history.clear()
        self._clear_state()
        self._schedule_evaluation()

    def make_human_move(self, move: chess.Move) -> bool:
        if not self.is_human_turn():
            return False
        if not self._push_move(move):
            return False
        self._apply_position_change()
        return True

    def make_ai_move(self) -> bool:
        if not self.is_ai_turn():
            return False
        move = self._current_engine().get_best_move(self.copy_board())
        self._push_move(move)
        self._apply_position_change()
        return True

    def _handle_ai_result(self):
        self.last_ai_time_ms = self.ai_worker.result_time_ms
        if self.ai_worker.result_error:
            return
        if self.ai_worker.result_fen != self.board.fen() or not self.is_ai_turn():
            return
        move = self.ai_worker.result_move or self._fallback_legal_move()
        if not move:
            return
        if not self._push_move(move):
            return
        self._apply_position_change()

    def update_ai_workers(self):
        self.sf_eval.get_eval_from_eval_thread()
        self.nn_eval.get_eval_from_eval_thread()

        if self.ai_worker.get_move_from_engine_thread():
            self._handle_ai_result()
            self.ai_worker.clear_result()

        board_copy = None
        if not self.sf_eval.busy and self.sf_eval.pending:
            board_copy = self._request_board(board_copy)
            self.sf_eval.try_request(board_copy)

        if not self.nn_eval.busy and self.nn_eval.pending:
            board_copy = self._request_board(board_copy)
            self.nn_eval.try_request(board_copy)

        if self.is_ai_turn():
            board_copy = self._request_board(board_copy)
            self.ai_worker.request(self._current_engine(), board_copy)

    def undo_last_turn(self) -> int:
        if self.mode == AI_VS_AI:
            return 0
        self.ai_worker.cancel()
        undone = self._undo_moves(1 if self.mode == PVP else 2)
        if undone:
            self._clear_state()
            self._schedule_evaluation()
        return undone

    def save_pgn(self):
        game = chess.pgn.Game()
        node = game
        for m in self.board.move_stack:
            node = node.add_variation(m)

        game.headers["Event"] = self.mode.upper()
        game.headers["Date"] = datetime.date.today().isoformat()
        game.headers["White"] = self._pgn_player_name(chess.WHITE)
        game.headers["Black"] = self._pgn_player_name(chess.BLACK)
        del game.headers["Site"]
        del game.headers["Round"]

        game.headers["Result"] = self.result()
        outcome = self.board.outcome()
        if outcome:
            game.headers["Termination"] = outcome.termination.name
            if outcome.winner is not None:
                game.headers["Winner"] = "White" if outcome.winner else "Black"

        game.headers["PlyCount"] = str(len(self.board.move_stack))

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn_text = game.accept(exporter)

        os.makedirs(os.path.dirname(AI_PGN_PATH), exist_ok=True)
        with open(AI_PGN_PATH, "a", encoding="utf-8") as f:
            f.write(pgn_text)
            f.write("\n\n")

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
