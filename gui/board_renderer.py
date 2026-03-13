import pygame
import chess

from config import *
from gui.theme import *

PROMOTION_PIECES = [
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT
]

class BoardRenderer:
    def __init__(self, piece_images):
        self.piece_images = piece_images

    def draw(self, screen, board, gui_state):
        self._draw_board(screen)
        self._draw_last_move(screen, board, gui_state)
        self._draw_selected_square(screen, gui_state)
        self._draw_legal_moves(screen, board, gui_state)
        self._draw_check_alert(screen, board, gui_state)
        self._draw_pieces(screen, board, gui_state)
        self._draw_drag_piece(screen, gui_state)
        self._draw_promotion(screen, gui_state, self.piece_images)


    def _square_to_screen(self, square, flipped):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if flipped:
            col, row = 7 - file, rank
        else:
            col, row = file, 7 - rank
        return col * SQUARE_SIZE, row * SQUARE_SIZE

    def _draw_board(self, screen):
        for rank in range(8):
            for file in range(8):
                color = LIGHT_SQUARE if (rank+file)%2==0 else DARK_SQUARE
                rect = pygame.Rect(file*SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(screen,color,rect)

    def _draw_last_move(self, screen, board, gui_state):
        if not board.move_stack:
            return

        move = board.move_stack[-1]
        for sq in (move.from_square, move.to_square):
            x, y = self._square_to_screen(sq, gui_state.board_flipped)
            rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, LAST_MOVE_COLOR, rect, 4)

    def _draw_selected_square(self, screen, gui_state):
        if gui_state.selected_square is None:
            return
        x, y = self._square_to_screen(gui_state.selected_square, gui_state.board_flipped)
        pygame.draw.rect(screen, MOVE_HINT_COLOR, pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE), 4)

    def _draw_legal_moves(self, screen, board, gui_state):
        for move in gui_state.legal_moves:
            sq = move.to_square
            x, y = self._square_to_screen(sq, gui_state.board_flipped)
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
            if board.piece_at(sq):
                pygame.draw.circle(screen, CHECK_ALERT_COLOR, center, SQUARE_SIZE // 2 - 8, 4)
            else:
                pygame.draw.circle(screen, MOVE_HINT_COLOR, center, 10)

    def _draw_check_alert(self, screen, board, gui_state):
        if not board.is_check():
            return
        king_square = board.king(board.turn)
        if king_square is None:
            return
        x, y = self._square_to_screen(king_square, gui_state.board_flipped)
        pygame.draw.rect(screen, CHECK_ALERT_COLOR, pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE), 5)

    def _draw_pieces(self, screen, board, gui_state):
        for square, piece in board.piece_map().items():
            if gui_state.dragging and square == gui_state.selected_square:
                continue

            x, y = self._square_to_screen(square, gui_state.board_flipped)
            screen.blit(self.piece_images[piece.symbol()], (x,y))

    def _draw_drag_piece(self, screen, gui_state):
        if not gui_state.dragging:
            return

        mx,my = pygame.mouse.get_pos()
        x = mx - gui_state.drag_offset[0]
        y = my - gui_state.drag_offset[1]
        screen.blit(self.piece_images[gui_state.drag_piece_symbol], (x,y))

    def _draw_promotion(self,screen, gui_state, piece_images):
        if not gui_state.pending_promotion:
            return

        sq = gui_state.pending_promotion.to_square
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)

        if gui_state.board_flipped:
            x = (7 - file) * SQUARE_SIZE
            y = rank * SQUARE_SIZE
        else:
            x = file * SQUARE_SIZE
            y = (7 - rank) * SQUARE_SIZE

        rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE*4)
        pygame.draw.rect(screen,PROMOTION_BG,rect)

        for i, piece in enumerate(PROMOTION_PIECES):
            cell = pygame.Rect(x, y + i*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen,PROMOTION_CELL,cell)
            symbol = chess.piece_symbol(piece)
            img = piece_images[symbol]
            screen.blit(img,cell.topleft)
