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
class EvalSummary:
    stockfish: EvalDisplay
    neural: EvalDisplay

@dataclass
class PlayerSummary:
    white: str
    black: str

@dataclass
class AIStatus:
    label: str
    thinking: bool
    thinking_time_ms: int
    last_move_time_ms: Optional[int]

@dataclass
class GameStatus:
    mode_label: str
    turn_label: str
    state_label: str
    players: PlayerSummary
    ai: AIStatus
    move_rows: list[tuple[str, str, str]]
    result_text: Optional[str]
    evals: EvalSummary


def _eval_fill(centipawns: Optional[int]) -> float:
    if centipawns is None:
        return 0.5
    return 0.5 + max(-1200, min(1200, centipawns)) / 2400

def _eval_label(centipawns: Optional[int]) -> str:
    if centipawns is None:
        return "--"
    if abs(centipawns) >= 99999:
        return "+M" if centipawns > 0 else "-M"
    return f"{centipawns / 100.0:+.1f}"

def _eval_display(tracker) -> EvalDisplay:
    return EvalDisplay(
        label=_eval_label(tracker.cp),
        fill=_eval_fill(tracker.cp),
        pending=tracker.pending,
        error=tracker.error,
    )

def _player_label(game, color: bool) -> str:
    if game.mode == PVP or (game.mode == PVE and color == game.human_color):
        return "Human"
    engine_name = AI_WHITE_ENGINE if color == chess.WHITE else AI_BLACK_ENGINE
    if engine_name == STOCKFISH_AI:
        return f"{engine_name.upper()} ({STOCKFISH_ELO} Elo)"
    return f"{engine_name.upper()} (d={AI_MINMAX_DEPTH})"

def _result_text(game) -> Optional[str]:
    if not game.board.is_game_over():
        return None
    outcome = game.board.outcome()
    if outcome is None:
        return game.board.result()
    if outcome.winner is chess.WHITE:
        prefix = "White wins"
    elif outcome.winner is chess.BLACK:
        prefix = "Black wins"
    else:
        prefix = "Draw"
    return f"{prefix} by {outcome.termination.name.replace('_', ' ').title()}"

def _move_rows(game) -> list[tuple[str, str, str]]:
    history = game.move_history
    return [
        (f"{i // 2 + 1}.", history[i], history[i + 1] if i + 1 < len(history) else "")
        for i in range(0, len(history), 2)
    ]

def _ai_status(game) -> AIStatus:
    thinking = game.ai_worker.busy and game.is_ai_turn()
    think_time_ms = game.ai_worker.current_think_time_ms()
    if thinking:
        label = f"AI thinking: {think_time_ms / 1000:.2f}s"
    elif game.ai_worker.busy:
        label = "Finishing previous search"
    elif game.last_ai_time_ms is not None:
        label = f"Last AI move: {game.last_ai_time_ms / 1000:.2f}s"
    else:
        label = "Idle"
    return AIStatus(label, thinking, think_time_ms, game.last_ai_time_ms)

def _state_label(game) -> str:
    if game.board.is_checkmate():
        return "Checkmate"
    if game.board.is_stalemate():
        return "Stalemate"
    if game.board.is_insufficient_material():
        return "Drawn endgame"
    if game.board.is_check():
        return "Check"
    if game.is_ai_turn():
        return "AI to move"
    if game.mode == PVE:
        return "Your turn"
    return "Waiting for move"


def build_game_status(game) -> GameStatus:
    return GameStatus(
        mode_label=game.mode.upper(),
        turn_label="White to move" if game.board.turn else "Black to move",
        state_label=_state_label(game),
        players=PlayerSummary(
            white=_player_label(game, chess.WHITE),
            black=_player_label(game, chess.BLACK),
        ),
        ai=_ai_status(game),
        move_rows=_move_rows(game),
        result_text=_result_text(game),
        evals=EvalSummary(
            stockfish=_eval_display(game.sf_eval),
            neural=_eval_display(game.nn_eval),
        ),
    )