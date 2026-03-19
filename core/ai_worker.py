from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any, Optional

import chess


class _SingleFutureWorker:
    def __init__(self, thread_name_prefix: str):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._future: Optional[Future] = None

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(self, fn, *args):
        self._future = self._executor.submit(fn, *args)

    def take_completed_future(self) -> Optional[Future]:
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        return future

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=False)


class AIMoveWorker(_SingleFutureWorker):
    def __init__(self):
        super().__init__("chess-ai")
        self._started_at = 0.0
        self._engine: Optional[Any] = None
        self._requested_fen: Optional[str] = None
        self.result_fen: Optional[str] = None
        self.result_move: Optional[chess.Move] = None
        self.error: Optional[str] = None
        self.result_time_ms = 0

    def _clear_request(self):
        self._engine = None
        self._requested_fen = None

    def request(self, engine: Any, board: chess.Board) -> bool:
        if self.busy:
            return False
        board_copy = board.copy(stack=True)
        self._started_at = time.perf_counter()
        self._requested_fen = board_copy.fen()
        self._engine = engine
        self.error = None
        self.submit(self.solve_move, engine, board_copy)
        return True

    def current_think_time_ms(self) -> int:
        return int((time.perf_counter() - self._started_at) * 1000) if self.busy else 0

    def get_move_from_engine_thread(self) -> bool:
        future = self.take_completed_future()
        if future is None:
            return False
        self.result_fen = self._requested_fen
        self.result_move, self.result_time_ms, self.error = future.result()
        return True

    @staticmethod
    def solve_move(engine: Any, board: chess.Board):
        start = time.perf_counter()
        try:
            move = engine.get_best_move(board)
            return move, int((time.perf_counter() - start) * 1000), None
        except Exception as exc:
            time.sleep(0.1)
            default_move = next(iter(board.legal_moves), None)
            return default_move, int((time.perf_counter() - start) * 1000), str(exc)

    def clear_result(self):
        self.result_fen = None
        self.result_move = None
        self.result_error = None
        self.result_time_ms = 0

    def cancel(self):
        if not self.busy or self._engine is None:
            return
        fn = getattr(self._engine, "cancel_search", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

class EvalTracker(_SingleFutureWorker):
    def __init__(self, engine):
        super().__init__("chess-eval")
        self.engine = engine
        self.cp: Optional[int] = None
        self.error: Optional[str] = None
        self.result_fen: Optional[str] = None
        self.pending_fen: Optional[str] = None

    @property
    def pending(self) -> bool:
        return self.busy or self.pending_fen != self.result_fen

    def schedule(self, fen: str):
        self.pending_fen = fen

    def clear(self):
        self.cp = self.error = self.result_fen = None

    def get_eval_from_eval_thread(self):
        future = self.take_completed_future()
        if future is None:
            return
        try:
            fen, cp, err = future.result()
        except Exception as exc:
            self.error = str(exc)
            return
        if err:
            self.error = err
            self.result_fen = fen
        elif fen == self.pending_fen:
            self.cp, self.error, self.result_fen = cp, None, fen

    def try_request(self, board: chess.Board):
        if self.busy or self.pending_fen == self.result_fen:
            return
        self.submit(self.evaluate, board.copy(stack=True))

    def evaluate(self, board: chess.Board):
        fen = board.fen()
        try:
            cp = int(round(self.engine.evaluate_position(board)))
            return fen, cp, None
        except Exception as exc:
            return fen, None, str(exc)