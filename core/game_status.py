from dataclasses import dataclass
from typing import Optional
import chess

from config import *


@dataclass
class EvalDisplay:
    label: str
    fill: float
    pending: bool
    error: Optional[str]

@dataclass
class GameStatus:
    mode_label: str
    turn_label: str
    state_label: str

    white_player: str
    black_player: str

    ai_label: str
    ai_thinking: bool

    move_rows: list[tuple[str, str, str]]

    result_text: Optional[str]

    stockfish_eval: EvalDisplay
    neural_eval: EvalDisplay


def _eval_display(tracker) -> EvalDisplay:
    cp = tracker.cp
    if cp is None:
        label = "--"
        fill = 0.5
    else:
        fill = 0.5 + max(-1200, min(1200, cp)) / 2400
        if abs(cp) >= 99999:
            label = "+M" if cp > 0 else "-M"
        else:
            label = f"{cp / 100:+.1f}"

    return EvalDisplay(label=label, fill=fill, pending=tracker.pending, error=tracker.error)

def _player_label(game, color):
    if game.mode == PVP or (game.mode == PVE and color == game.human_color):
        return "Human"
    engine = AI_WHITE_ENGINE if color == chess.WHITE else AI_BLACK_ENGINE
    if engine == STOCKFISH_AI:
        return f"{engine.upper()} ({STOCKFISH_ELO} Elo)"
    return f"{engine.upper()} (d={AI_MINMAX_DEPTH})"

def _state_label(game):
    board = game.board
    if board.is_checkmate():
        return "Checkmate"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return "Draw"
    if board.is_check():
        return "Check"
    if game.is_ai_turn():
        return "AI to move"
    if game.mode == PVE:
        return "Your turn"
    return ""

def _result_text(game):
    board = game.board
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome is None:
        return board.result()
    if outcome.winner is chess.WHITE:
        prefix = "White wins"
    elif outcome.winner is chess.BLACK:
        prefix = "Black wins"
    else:
        prefix = "Draw"
    return f"{prefix} by {outcome.termination.name.replace('_', ' ').title()}"

def _move_rows(game):
    rows = []
    history = game.move_history
    for i in range(0, len(history), 2):
        white = history[i]
        black = history[i + 1] if i + 1 < len(history) else ""
        rows.append((f"{i//2+1}.", white, black))
    return rows

def _ai_status(game):
    worker = game.ai_worker
    thinking = worker.busy and game.is_ai_turn()

    if bool(worker.error):
        label = f"Eval unavailable, AI moves randomly"
    elif thinking:
        t = worker.current_think_time_ms() / 1000
        label = f"AI thinking: {t:.2f}s"
    elif worker.busy:
        label = "Finishing previous search"
    elif game.last_ai_time_ms is not None:
        t = game.last_ai_time_ms / 1000
        label = f"Last AI move: {t:.2f}s"
    else:
        label = "Idle"
    return label, thinking


def build_game_status(game) -> GameStatus:
    ai_label, ai_thinking = _ai_status(game)
    return GameStatus(
        mode_label=game.mode.upper(),
        turn_label="White to move" if game.board.turn else "Black to move",
        state_label=_state_label(game),
        white_player=_player_label(game, chess.WHITE),
        black_player=_player_label(game, chess.BLACK),
        ai_label=ai_label,
        ai_thinking=ai_thinking,
        move_rows=_move_rows(game),
        result_text=_result_text(game),
        stockfish_eval=_eval_display(game.sf_eval),
        neural_eval=_eval_display(game.nn_eval),
    )
