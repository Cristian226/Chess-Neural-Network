import chess
import torch


def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        planes[idx, row, col] = 1.0

    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    planes[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        planes[17, ep_row, ep_col] = 1.0

    return planes