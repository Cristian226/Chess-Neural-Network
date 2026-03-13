from dataclasses import dataclass, field
from typing import Optional
import chess


@dataclass
class PendingPromotion:
    from_square: int
    to_square: int


@dataclass
class GUIState:
    selected_square: Optional[int] = None
    legal_moves: list[chess.Move] = field(default_factory=list)

    dragging: bool = False
    drag_piece_symbol: Optional[str] = None
    drag_offset: tuple[int, int] = (0, 0)
    mouse_down_pos: Optional[tuple[int, int]] = None

    pending_promotion: Optional[PendingPromotion] = None
    board_flipped: bool = False
    move_scroll_offset: int = 0
    seen_move_count: int = 0

    def clear_selection(self):
        self.selected_square = None
        self.legal_moves = []
        self.dragging = False
        self.drag_piece_symbol = None
        self.drag_offset = (0, 0)

    def reset_interaction(self):
        self.clear_selection()
        self.mouse_down_pos = None

    def reset_ui(self, move_count: int):
        self.reset_interaction()
        self.pending_promotion = None
        self.move_scroll_offset = move_count
        self.seen_move_count = move_count

    def start_drag(self, square: int, legal_moves: list[chess.Move], symbol: str, offset: tuple[int, int]):
        self.selected_square = square
        self.legal_moves = legal_moves
        self.dragging = True
        self.drag_piece_symbol = symbol
        self.drag_offset = offset
