import pygame
import chess
from config import FPS, HEIGHT, WIDTH
from core.game_manager import GameManager
from core.game_status import build_game_status
from gui.renderer import Renderer
from gui.ui_state import GUIState, PendingPromotion

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")

        self.clock = pygame.time.Clock()
        self.manager = GameManager()
        self.ui_state = GUIState()
        self.renderer = Renderer(self.screen)
        self.running = True

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            
            self.handle_events()
            self.manager.update_ai_workers()
            self.update_gui()            

        self.manager.close()
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key_down(event.key)
            elif event.type == pygame.MOUSEWHEEL:
                self.handle_mouse_wheel(event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.ui_state.pending_promotion:
                    self.handle_promotion_click(event.pos)
                else:
                    self.handle_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_up(event.pos)

    def update_gui(self):
        status = build_game_status(self.manager)
        self._sync_move_scroll(self.manager.move_row_count,
                               self.renderer.panel_layout.visible_move_rows,
                               len(self.manager.move_history))
        self.renderer.draw(self.manager, self.ui_state, status)

    def handle_mouse_down(self, pos):
        square = self.renderer.square_from_mouse(pos, self.ui_state.board_flipped)
        if square is None or not self.manager.is_human_turn():
            return

        if self.ui_state.selected_square == square:
            self._reset_interaction()
            return

        if self.ui_state.selected_square is not None:
            if any(m.to_square == square for m in self.ui_state.legal_moves):
                self._try_move(self.ui_state.selected_square, square)
                return

        piece = self.manager.board.piece_at(square)
        if piece is None or piece.color != self.manager.board.turn:
            self._clear_selection()
            return

        legal_moves = self.manager.legal_moves_from(square)
        if not legal_moves:
            self._clear_selection()
            return

        rect = self.renderer.square_rect(square, self.ui_state.board_flipped)
        self._start_drag(square, legal_moves, piece.symbol(), (pos[0] - rect.x, pos[1] - rect.y))

    def handle_mouse_up(self, pos):
        if not self.ui_state.dragging or self.ui_state.selected_square is None:
            return

        square = self.renderer.square_from_mouse(pos, self.ui_state.board_flipped)
        from_sq = self.ui_state.selected_square
        self._clear_drag()

        if square is None:
            self._clear_selection()
            return

        if square == from_sq:
            return

        self._try_move(from_sq, square)

    def handle_promotion_click(self, pos):
        piece_type = self.renderer.promotion_piece_from_mouse(pos)
        if piece_type is None or self.ui_state.pending_promotion is None:
            return
        pending = self.ui_state.pending_promotion
        self.ui_state.pending_promotion = None
        self._submit_move(chess.Move(pending.from_square, pending.to_square, promotion=piece_type))

    def handle_key_down(self, key: int):
        if key == pygame.K_r:
            self.manager.reset()
            self._reset_ui()
        elif key == pygame.K_f:
            self.ui_state.board_flipped = not self.ui_state.board_flipped
        elif key == pygame.K_m:
            self.manager.cycle_mode()
            self._reset_ui()
        elif key == pygame.K_u:
            if self.manager.undo_last_turn():
                self._reset_ui()

    def handle_mouse_wheel(self, delta_y: int):
        mouse_pos = pygame.mouse.get_pos()
        if self.renderer.panel_layout.moves_viewport.collidepoint(mouse_pos):
            self.ui_state.move_scroll_offset = max(0, self.ui_state.move_scroll_offset - delta_y)

    def _try_move(self, from_square: int, to_square: int):
        if any( m.to_square == to_square and m.promotion is not None for m in self.manager.legal_moves_from(from_square)):
            self.ui_state.pending_promotion = PendingPromotion(from_square, to_square)
            return
        self._submit_move(chess.Move(from_square, to_square))

    def _submit_move(self, move: chess.Move):
        self.manager.make_human_move(move)
        self._clear_selection()

    def _start_drag(self, square: int, legal_moves, piece_symbol: str, drag_offset):
        self.ui_state.selected_square = square
        self.ui_state.legal_moves = legal_moves
        self.ui_state.dragging = True
        self.ui_state.drag_piece_symbol = piece_symbol
        self.ui_state.drag_offset = drag_offset

    def _clear_drag(self):
        self.ui_state.dragging = False
        self.ui_state.drag_piece_symbol = None
        self.ui_state.drag_offset = (0, 0)

    def _clear_selection(self):
        self.ui_state.selected_square = None
        self.ui_state.legal_moves = []

    def _reset_interaction(self):
        self._clear_drag()
        self._clear_selection()
        self.ui_state.pending_promotion = None

    def _reset_move_scroll(self):
        self.ui_state.move_scroll_offset = 0

    def _sync_move_scroll(self, total_rows: int, visible_rows: int, move_count: int):
        max_start = max(0, total_rows - visible_rows)
        if move_count > self.ui_state.seen_move_count:
            self.ui_state.move_scroll_offset = max_start
        else:
            self.ui_state.move_scroll_offset = min(self.ui_state.move_scroll_offset, max_start)
        self.ui_state.seen_move_count = move_count

    def _reset_ui(self):
        self._reset_interaction()
        self._reset_move_scroll()
