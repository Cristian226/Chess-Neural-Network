import pygame
import chess
from config import *

class Renderer:
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        self.screen = screen
        self.font = font
        self.load_assets()

    def load_assets(self):
        self.pieces = {}
        mapping = {
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
        }
        for k, v in mapping.items():
            self.pieces[k] = pygame.transform.scale(pygame.image.load(f"{PIECE_PATH}{v}.png"), (SQ_SIZE, SQ_SIZE))

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BROWN
                pygame.draw.rect(
                    self.screen,
                    color,
                    (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )

    def highlight_last_move(self, game):
        move = game.last_move
        if not move:
            return
        for sq in (move.from_square, move.to_square):
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            pygame.draw.rect(
                self.screen,
                LAST_MOVE,
                (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )

    def highlight_selected(self, selected_square):
        if selected_square is None:
            return
        col = chess.square_file(selected_square)
        row = 7 - chess.square_rank(selected_square)
        pygame.draw.rect(
            self.screen,
            SELECTED,
            (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            4
        )

    def highlight_moves(self, legal_moves):
        for move in legal_moves:
            col = chess.square_file(move.to_square)
            row = 7 - chess.square_rank(move.to_square)
            pygame.draw.rect(
                self.screen,
                HIGHLIGHT,
                (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                3
            )

    def draw_pieces(self, game, dragging, selected_square, drag_piece, drag_offset):
        for sq in chess.SQUARES:
            if dragging and sq == selected_square:
                continue

            piece = game.board.piece_at(sq)
            if piece:
                col = chess.square_file(sq)
                row = 7 - chess.square_rank(sq)
                self.screen.blit(self.pieces[piece.symbol()], (col * SQ_SIZE, row * SQ_SIZE))

        if dragging and drag_piece:
            mx, my = pygame.mouse.get_pos()
            self.screen.blit(drag_piece, (mx - drag_offset[0], my - drag_offset[1]))

    def draw_status(self, game):
        pygame.draw.rect(self.screen, BLACK, (0, BOARD_SIZE, WIDTH, HEIGHT - BOARD_SIZE))

        if game.board.is_checkmate():
            text = "Checkmate! Press R to restart"
        elif game.board.is_check():
            text = "Check!"
        else:
            text = "White to move" if game.board.turn else "Black to move"

        mode_text = f"Mode: {GAME_MODE.upper()}"
        label = self.font.render(text, True, RED if "Check" in text else WHITE)
        mode_label = self.font.render(mode_text, True, WHITE)
        self.screen.blit(label, (10, BOARD_SIZE + 15))
        self.screen.blit(mode_label, (WIDTH - 10 - mode_label.get_width(), BOARD_SIZE + 15))

    def draw_promotion_menu(self, pending_promotion, game):
        if not pending_promotion:
            return
        pygame.draw.rect(self.screen, GRAY, (160, 250, 320, 120))
        symbols = ['Q', 'R', 'B', 'N'] if game.board.turn else ['q', 'r', 'b', 'n']
        for i, sym in enumerate(symbols):
            self.screen.blit(
                self.pieces[sym],
                (180 + i * 70, 270)
            )

    def draw_all(self, game, selected_square, legal_moves, dragging, drag_piece, drag_offset, pending_promotion):
        self.draw_board()
        self.highlight_last_move(game)
        self.highlight_selected(selected_square)
        self.highlight_moves(legal_moves)
        self.draw_pieces(game, dragging, selected_square, drag_piece, drag_offset)
        self.draw_status(game)

        if pending_promotion:
            self.draw_promotion_menu(pending_promotion, game)

        pygame.display.flip()
