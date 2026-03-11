from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import chess


@dataclass
class PendingPromotion:
    from_square: int
    to_square: int


@dataclass
class GUIState:
    selected_square: Optional[int] = None
    legal_moves: List[chess.Move] = field(default_factory=list)
    dragging: bool = False
    drag_piece_symbol: Optional[str] = None
    drag_offset: Tuple[int, int] = (0, 0)
    pending_promotion: Optional[PendingPromotion] = None
    board_flipped: bool = False
    move_scroll_offset: int = 0
    seen_move_count: int = 0