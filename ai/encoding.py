import chess
import numpy as np
import torch

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


def bb_to_plane(bb: int) -> torch.Tensor:
    raw  = np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little").astype(np.float32)
    return torch.from_numpy(bits.copy()).view(8, 8).flip(0)

def simple_score(board: chess.Board) -> float:
    score = 0
    for piece_type, weight in PIECE_SCORES.items():
        score += weight * (
            len(board.pieces(piece_type, chess.WHITE)) -
            len(board.pieces(piece_type, chess.BLACK))
        )
    return score / MAX_MATERIAL

def encode_board(board: chess.Board) -> torch.Tensor:
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
    board = board.copy()
    if board.turn == chess.BLACK:
        board = board.mirror()

    planes = torch.zeros((20, 8, 8), dtype=torch.float32)

    for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for i, pt in enumerate(PIECE_TYPES):
            bb = int(board.pieces_mask(pt, color))
            if bb:
                planes[offset + i] = bb_to_plane(bb)

    white_att = black_att = 0
    for sq in chess.SquareSet(board.occupied):
        bb = int(board.attacks_mask(sq))
        if board.color_at(sq) == chess.WHITE:
            white_att |= bb
        else:
            black_att |= bb

    if white_att:
        planes[12] = bb_to_plane(white_att)
    if black_att:
        planes[13] = bb_to_plane(black_att)

    planes[14].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0)
    planes[15].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[16].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0)
    planes[17].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    if board.ep_square is not None:
        planes[18] = bb_to_plane(chess.BB_SQUARES[board.ep_square])

    planes[19].fill_(simple_score(board))

    return planes