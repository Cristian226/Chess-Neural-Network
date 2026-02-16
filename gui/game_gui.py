import pygame
import chess
from config import BOARD_SIZE, FONT_SIZE, FPS, HEIGHT, SQ_SIZE, WIDTH
from core.game_manager import GameManager
from gui.renderer import Renderer

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.manager = GameManager()
        self.game = self.manager.game

        self.selected_square = None
        self.legal_moves = []
        self.dragging = False
        self.drag_piece = None
        self.drag_offset = (0, 0)
        self.pending_promotion = None

        # Renderer handles drawing and asset loading
        self.renderer = Renderer(self.screen, self.font)

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.handle_key_reset()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.pending_promotion:
                        self.handle_click(event.pos)
                    else:
                        self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos)

            self.manager.update(pygame.time.get_ticks())
            self.renderer.draw_all(self.game, self.selected_square, self.legal_moves, self.dragging, self.drag_piece, self.drag_offset, self.pending_promotion)

        pygame.quit()


    def handle_mouse_down(self, pos):
        square = self.square_from_mouse(pos)
        if square is None:
            return
        if not self.manager.is_human_turn():
            return
        piece = self.game.board.piece_at(square)
        if piece and ((piece.color == chess.WHITE and self.game.board.turn) or (piece.color == chess.BLACK and not self.game.board.turn)):
            self.selected_square = square
            self.legal_moves = [m for m in self.game.board.legal_moves if m.from_square == square]
            self.dragging = True
            self.drag_piece = self.renderer.pieces[piece.symbol()]
            mx, my = pos
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            self.drag_offset = (mx - col*SQ_SIZE, my - row*SQ_SIZE)

    def handle_mouse_up(self, pos):
        if not self.dragging or self.selected_square is None:
            return

        square = self.square_from_mouse(pos)
        if square is None:
            self.selected_square = None
            self.dragging = False
            self.drag_piece = None
            return

        piece = self.game.board.piece_at(self.selected_square)
        if piece.piece_type == chess.PAWN and chess.square_rank(square) in (0,7):
            self.pending_promotion = (self.selected_square, square)
        else:
            move = chess.Move(self.selected_square, square)
            now = pygame.time.get_ticks()
            self.manager.make_human_move(move, now)

        self.selected_square = None
        self.legal_moves = []
        self.dragging = False
        self.drag_piece = None

    def handle_click(self, pos):
        # Only for promotion menu
        x, y = pos
        if 180 <= x <= 460 and 270 <= y <= 340:
            index = (x - 180) // 70
            promo_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][index]
            move = chess.Move(self.pending_promotion[0], self.pending_promotion[1], promotion=promo_piece)
            now = pygame.time.get_ticks()
            self.manager.make_human_move(move, now)
            self.pending_promotion = None

    def handle_key_reset(self):
        self.manager.reset()
        self.selected_square = None
        self.legal_moves = []
        self.pending_promotion = None
        self.dragging = False
        self.drag_piece = None

    def square_from_mouse(self, pos):
        x, y = pos
        if y >= BOARD_SIZE:
            return None
        col = x // SQ_SIZE
        row = 7 - (y // SQ_SIZE)
        return chess.square(col, row)