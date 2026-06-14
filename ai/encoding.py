from __future__ import annotations
import threading
import chess
import torch

_RANK_BITS: list[torch.Tensor] = [
    torch.tensor([(b >> bit) & 1 for bit in range(8)], dtype=torch.float32)
    for b in range(256)
]

# Per thread scratch buffer: encode_board is called concurrently from the AI search thread and the eval thread
_local = threading.local()

def _get_buf() -> torch.Tensor:
    buf = getattr(_local, "buf", None)
    if buf is None:
        buf = torch.zeros(20, 8, 8, dtype=torch.float32)
        _local.buf = buf
    return buf

PIECE_SCORES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
MAX_MATERIAL = 39.0

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _write_bb(bb: int, plane: torch.Tensor) -> None:
    raw = bb.to_bytes(8, "little")
    for rank in range(8):
        plane[7 - rank] = _RANK_BITS[raw[rank]]

def simple_score(board: chess.Board) -> float:
    score = 0
    for pt, w in PIECE_SCORES.items():
        score += w * (
            chess.popcount(board.pieces_mask(pt, chess.WHITE)) -
            chess.popcount(board.pieces_mask(pt, chess.BLACK))
        )
    return score / MAX_MATERIAL


def encode_board(board: chess.Board, clone: bool = True) -> torch.Tensor:
    """
    0–5     White pieces
    6–11    Black pieces
    12      Squares attacked by White
    13      Squares attacked by Black
    14      White kingside castling right
    15      White queenside castling right
    16      Black kingside castling right
    17      Black queenside castling right
    18      En-passant target square
    19      Simple score [-1, 1]
    """
    if board.turn == chess.BLACK:
        board = board.mirror()

    buf = _get_buf()
    buf.zero_()

    for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for i, pt in enumerate(PIECE_TYPES):
            bb = int(board.pieces_mask(pt, color))
            if bb:
                _write_bb(bb, buf[offset + i])

    white_att = 0
    for sq in chess.SquareSet(board.occupied_co[chess.WHITE]):
        white_att |= int(board.attacks_mask(sq))
    if white_att:
        _write_bb(white_att, buf[12])

    black_att = 0
    for sq in chess.SquareSet(board.occupied_co[chess.BLACK]):
        black_att |= int(board.attacks_mask(sq))
    if black_att:
        _write_bb(black_att, buf[13])

    buf[14].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0)
    buf[15].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    buf[16].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0)
    buf[17].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    if board.ep_square is not None:
        _write_bb(int(chess.BB_SQUARES[board.ep_square]), buf[18])

    buf[19].fill_(simple_score(board))

    return buf.clone() if clone else buf