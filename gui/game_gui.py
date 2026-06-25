import pygame
import chess

from config import *
from core.game_manager import GameManager
from gui.renderer import Renderer
from gui.ui_state import GUIState, PendingPromotion
from core.game_status import build_game_status


class GameGUI:
    def __init__(self, manager: GameManager):
        pygame.init()
        self.manager = manager
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer()
        self.gui_state = GUIState()
        self.running = True

    def run(self):
        while self.running:
            self.clock.tick(144)
            self.manager.update_ai_workers()
            self._handle_events()
            self.renderer.draw(self.screen, self.manager, self.gui_state, build_game_status(self.manager))
            self._sync_move_scroll(self.manager.move_row_count, self.renderer.panel_layout.visible_move_rows)

        self.manager.close()
        pygame.quit()


    def square_from_mouse(self, pos, flipped):
        x, y = pos
        if x >= BOARD_SIZE:
            return None
        
        file = x // SQUARE_SIZE
        rank_from_top = y // SQUARE_SIZE
        
        if flipped:
            file = 7 - file
            rank = rank_from_top
        else:
            rank = 7 - rank_from_top
        
        return chess.square(file, rank)

    def square_rect(self, square, flipped):
        x, y = self.renderer.board_renderer._square_to_screen(square, flipped)
        return pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)


    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_key_down(event.key)
            elif event.type == pygame.MOUSEWHEEL:
                self._on_mouse_wheel(event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._on_mouse_up(event)

    def _on_mouse_down(self, event):
        pos = event.pos
        self.gui_state.mouse_down_pos = pos

        piece = self._promotion_click(self.gui_state, pos)
        if piece:
            pp = self.gui_state.pending_promotion
            self.manager.try_move(pp.from_square, pp.to_square, piece)
            self.gui_state.pending_promotion = None
            return

        square = self.square_from_mouse(pos, self.gui_state.board_flipped)
        if square is None or self.manager.is_ai_turn():
            return

        if self.gui_state.selected_square == square:
            self.gui_state.reset_interaction()
            return

        if self.gui_state.selected_square is not None:
            if any(m.to_square == square for m in self.gui_state.legal_moves):
                self._try_move(self.gui_state.selected_square, square)
                return

        piece = self.manager.board.piece_at(square)
        if piece is None or piece.color != self.manager.board.turn:
            self.gui_state.clear_selection()
            return

        legal_moves = self.manager.legal_moves_from(square)
        if not legal_moves:
            self.gui_state.clear_selection()
            return

        rect = self.square_rect(square, self.gui_state.board_flipped)
        self.gui_state.start_drag(square, legal_moves, piece.symbol(), (pos[0] - rect.x, pos[1] - rect.y))

    def _on_mouse_up(self, event):
        if not self.gui_state.dragging:
            return

        pos = event.pos
        if self.gui_state.mouse_down_pos:
            dx = pos[0] - self.gui_state.mouse_down_pos[0]
            dy = pos[1] - self.gui_state.mouse_down_pos[1]
            moved = (dx * dx + dy * dy) > DRAG_DISTANCE_THRESHOLD_SQ
        else:
            return
        if not moved:
            return

        to_sq = self.square_from_mouse(pos, self.gui_state.board_flipped)
        from_sq = self.gui_state.selected_square

        if to_sq is not None and from_sq is not None:
            if any(m.to_square == to_sq for m in self.gui_state.legal_moves):
                self._try_move(from_sq, to_sq)
                return

        self.gui_state.reset_interaction()

    def _on_key_down(self, key):
        if key == pygame.K_r:
            self.manager.reset()
            self.gui_state.reset_ui(len(self.manager.move_history))
        elif key == pygame.K_f:
            self.gui_state.board_flipped = not self.gui_state.board_flipped
        elif key == pygame.K_u:
            if self.manager.undo_last_turn():
                self.gui_state.reset_ui(len(self.manager.move_history))
        elif key == pygame.K_m:
            self.manager.cycle_mode()
            self.gui_state.reset_ui(len(self.manager.move_history))

    def _on_mouse_wheel(self, delta_y):
        mouse_pos = pygame.mouse.get_pos()
        if self.renderer.panel_layout.moves_viewport.collidepoint(mouse_pos):
            self.gui_state.move_scroll_offset = max(0, self.gui_state.move_scroll_offset - delta_y)


    def _try_move(self, from_sq, to_sq):
        piece = self.manager.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_sq)
            if rank in (0, 7):
                self.gui_state.pending_promotion = PendingPromotion(from_sq, to_sq)
                self.gui_state.reset_interaction()
                return

        self.manager.try_move(from_sq, to_sq)
        self.gui_state.reset_interaction()

    def _sync_move_scroll(self, total_rows, visible_rows):
        max_start = max(0, total_rows - visible_rows)
        move_count = len(self.manager.move_history)
        if move_count > self.gui_state.seen_move_count:
            self.gui_state.move_scroll_offset = max_start
        else:
            self.gui_state.move_scroll_offset = min(self.gui_state.move_scroll_offset, max_start)
        self.gui_state.seen_move_count = move_count

    def _promotion_click(self, gui_state, pos):
        if not gui_state.pending_promotion:
            return None

        _, cells = self.renderer.board_renderer.promotion_layout(gui_state.pending_promotion.to_square, gui_state.board_flipped)
        for piece, cell in cells:
            if cell.collidepoint(pos):
                return piece
        return None
