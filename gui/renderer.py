import pygame
import chess
from dataclasses import dataclass
from config import *


@dataclass
class PanelLayout:
    panel: pygame.Rect
    content: pygame.Rect
    gap: int
    header: pygame.Rect
    notice: pygame.Rect
    stockfish: pygame.Rect
    neural: pygame.Rect
    moves: pygame.Rect
    moves_viewport: pygame.Rect
    visible_move_rows: int
    game: pygame.Rect
    engine: pygame.Rect
    eval_bar: pygame.Rect


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.small_font = pygame.font.SysFont(None, SMALL_FONT_SIZE)
        self.panel_layout = self._create_panel_layout()
        self.board_surface = self._create_board_surface()
        self._load_assets()

    def _load_assets(self):
        self.pieces = {}
        for symbol in 'PNBRQKpnbrqk':
            name = ('w' if symbol.isupper() else 'b') + symbol.lower()
            image = pygame.image.load(f"{PIECE_PATH}{name}.png").convert_alpha()
            self.pieces[symbol] = pygame.transform.smoothscale(image, (SQ_SIZE, SQ_SIZE))

    def _create_board_surface(self) -> pygame.Surface:
        surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BROWN
                pygame.draw.rect(surface, color, (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
        return surface


    def square_rect(self, square: int, flipped: bool) -> pygame.Rect:
        x, y = self._square_to_screen(square, flipped)
        return pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)

    def _square_to_screen(self, square: int, flipped: bool) -> tuple[int, int]:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if flipped:
            col, row = 7 - file_idx, rank_idx
        else:
            col, row = file_idx, 7 - rank_idx
        return col * SQ_SIZE, row * SQ_SIZE

    def square_from_mouse(self, pos: tuple[int, int], flipped: bool) -> int | None:
        x, y = pos
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return None
        col, row = x // SQ_SIZE, y // SQ_SIZE
        if flipped:
            file_idx, rank_idx = 7 - col, row
        else:
            file_idx, rank_idx = col, 7 - row
        return chess.square(file_idx, rank_idx)


    def _draw_board_overlays(self, board: chess.Board, last_move, ui_state):
        flipped = ui_state.board_flipped

        if last_move:
            for sq in (last_move.from_square, last_move.to_square):
                pygame.draw.rect(self.screen, LAST_MOVE, self.square_rect(sq, flipped))

        if ui_state.selected_square is not None:
            pygame.draw.rect(self.screen, SELECTED, self.square_rect(ui_state.selected_square, flipped), 4)

        for move in ui_state.legal_moves:
            rect = self.square_rect(move.to_square, flipped)
            if board.piece_at(move.to_square):
                pygame.draw.circle(self.screen, CAPTURE_HINT, rect.center, SQ_SIZE // 2 - 8, 5)
            else:
                pygame.draw.circle(self.screen, MOVE_DOT, rect.center, 10)

        if board.is_check():
            king_sq = board.king(board.turn)
            if king_sq is not None:
                pygame.draw.rect(self.screen, CHECK_ALERT, self.square_rect(king_sq, flipped), 5)

    def _draw_pieces(self, board: chess.Board, ui_state):
        flipped = ui_state.board_flipped
        for sq in chess.SQUARES:
            if ui_state.dragging and sq == ui_state.selected_square:
                continue
            piece = board.piece_at(sq)
            if piece:
                self.screen.blit(self.pieces[piece.symbol()], self._square_to_screen(sq, flipped))

        if ui_state.dragging and ui_state.drag_piece_symbol:
            mx, my = pygame.mouse.get_pos()
            self.screen.blit(
                self.pieces[ui_state.drag_piece_symbol],
                (mx - ui_state.drag_offset[0], my - ui_state.drag_offset[1]),
            )


    def _draw_board(self, game, ui_state):
        self.screen.blit(self.board_surface, (0, 0))
        self._draw_board_overlays(game.board, game.last_move, ui_state)
        self._draw_pieces(game.board, ui_state)


    def _draw_eval_bar(self, status):
        bar = self.panel_layout.eval_bar
        ev = status.evals.stockfish
        pygame.draw.rect(self.screen, BLACK, bar, border_radius=10)

        white_h = int(round(bar.height * ev.fill))
        if white_h > 0:
            pygame.draw.rect(self.screen, TEXT_PRIMARY,
                             pygame.Rect(bar.x, bar.bottom - white_h, bar.width, white_h),
                             border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, bar, width=2, border_radius=10)

        handle_y = max(bar.y, bar.bottom - white_h - 3)
        handle = pygame.Rect(bar.x - 4, handle_y, bar.width + 8, 6)
        pygame.draw.rect(self.screen, PANEL_ACCENT if not ev.pending else WARNING, handle, border_radius=3)

        self.screen.blit(self.small_font.render("B", True, TEXT_MUTED), (bar.x + 5, bar.y - 18))
        self.screen.blit(self.small_font.render("W", True, TEXT_MUTED), (bar.x + 4, bar.bottom + 4))

    def _create_panel_layout(self) -> PanelLayout:
        panel = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)
        margin_x = max(14, panel.width // 18)
        margin_y = max(16, panel.height // 30)
        gap = max(10, panel.height // 48)
        eval_bar_w = max(18, min(26, panel.width // 12))
        eval_gap = max(10, panel.width // 24)
        content = pygame.Rect(panel.x + margin_x, panel.y + margin_y,
            panel.width - margin_x * 2 - eval_bar_w - eval_gap, panel.height - margin_y * 2)

        # Vertical slot heights
        header_h = max(56, int(content.height * 0.11))
        notice_h = max(18, int(content.height * 0.04))
        eval_h = max(72, int(content.height * 0.12))
        info_h = max(62, int(content.height * 0.10))
        engine_h = max(72, int(content.height * 0.12))
        moves_h = max(120, content.height - header_h - notice_h - eval_h * 2 - info_h - engine_h - gap * 6)

        # Stack cards top to bottom
        x, y = content.x, content.y
        header = pygame.Rect(x, y, content.width, header_h); y += header_h + gap
        notice = pygame.Rect(x, y, content.width, notice_h); y += notice_h + gap
        stockfish = pygame.Rect(x, y, content.width, eval_h); y += eval_h + gap
        neural = pygame.Rect(x, y, content.width, eval_h); y += eval_h + gap
        moves = pygame.Rect(x, y, content.width, moves_h); y += moves_h + gap
        game = pygame.Rect(x, y, content.width, info_h); y += info_h + gap
        engine = pygame.Rect(x, y, content.width, content.bottom - y)
        eval_bar = pygame.Rect(content.right + eval_gap, content.y, eval_bar_w, content.height)

        # Moves list inner viewport
        pad_x = max(8, moves.width // 26)
        hdr_h = max(28, moves.height // 5)
        scroll_w = max(6, moves.width // 42)
        moves_vp = pygame.Rect(
            moves.x + pad_x, moves.y + hdr_h,
            moves.width - pad_x * 2 - scroll_w - max(6, gap // 2),
            moves.height - hdr_h - max(10, gap),
        )

        return PanelLayout(panel, content, gap, header, notice, stockfish, neural, moves, moves_vp,
            max(1, moves_vp.height // 24), game, engine, eval_bar)


    def _draw_card(self, rect: pygame.Rect, title: str):
        pygame.draw.rect(self.screen, CARD_BG, rect, border_radius=14)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=2, border_radius=14)
        pad = max(12, rect.width // 20)
        self.screen.blit(self.small_font.render(title, True, TEXT_MUTED),
                         (rect.x + pad, rect.y + max(10, rect.height // 9)))

    def _draw_inline_eval_bar(self, rect: pygame.Rect, fill: float, pending: bool):
        pygame.draw.rect(self.screen, BLACK, rect, border_radius=8)
        w = int(round(rect.width * fill))
        if w > 0:
            pygame.draw.rect(self.screen, TEXT_PRIMARY, (rect.x, rect.y, w, rect.height), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=8)
        marker = pygame.Rect(rect.x + max(0, w - 2), rect.y - 2, 4, rect.height + 4)
        pygame.draw.rect(self.screen, PANEL_ACCENT if not pending else WARNING, marker, border_radius=2)

    def _draw_eval_card(self, rect: pygame.Rect, title: str, ev, accent: tuple):
        self._draw_card(rect, title)
        pad = max(12, rect.width // 20)
        color = accent if ev.pending else TEXT_PRIMARY
        text = f"{ev.label} ..." if ev.pending else ev.label
        self.screen.blit(self.font.render(text, True, color),
                         (rect.x + pad, rect.y + max(28, rect.height // 2 - 6)))
        bar = pygame.Rect(rect.x + pad, rect.bottom - max(18, rect.height // 4),
                          rect.width - pad * 2, max(8, rect.height // 9))
        self._draw_inline_eval_bar(bar, ev.fill, ev.pending)
        if ev.error:
            err = self.small_font.render("eval unavailable", True, CHECK_ALERT)
            self.screen.blit(err, (rect.right - err.get_width() - pad, rect.y + max(10, rect.height // 8)))


    def _draw_moves_list(self, status, scroll_offset: int):
        layout = self.panel_layout
        viewport = layout.moves_viewport
        rows = status.move_rows
        row_h = 24
        visible = layout.visible_move_rows
        max_start = max(0, len(rows) - visible)
        start = min(scroll_offset, max_start)

        self._draw_card(layout.moves, "Moves")
        pygame.draw.rect(self.screen, MOVES_BG, viewport, border_radius=10)

        num_x = viewport.x + max(6, viewport.width // 36)
        white_x = viewport.x + max(34, viewport.width // 6)
        black_x = viewport.x + max(92, viewport.width // 2 + viewport.width // 12)

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)
        y = viewport.y + 6
        for i, (num, w_move, b_move) in enumerate(rows[start:start + visible]):
            row_rect = pygame.Rect(viewport.x + 2, y - 1, viewport.width - 4, row_h - 2)
            if i % 2 == 0:
                pygame.draw.rect(self.screen, MOVES_ROW_ALT, row_rect, border_radius=8)
            self.screen.blit(self.small_font.render(num, True, TEXT_MUTED), (num_x, y + 2))
            self.screen.blit(self.small_font.render(w_move, True, TEXT_PRIMARY), (white_x, y + 2))
            self.screen.blit(self.small_font.render(b_move, True, TEXT_PRIMARY), (black_x, y + 2))
            y += row_h
        self.screen.set_clip(prev_clip)

        # Scrollbar
        if len(rows) > visible:
            track = pygame.Rect(layout.moves.right - max(10, layout.moves.width // 24),
                                viewport.y + 6, max(6, layout.moves.width // 42), viewport.height - 12)
            pygame.draw.rect(self.screen, SCROLLBAR_TRACK, track, border_radius=3)
            thumb_h = max(28, int(track.height * (visible / len(rows))))
            ratio = start / max(1, max_start)
            thumb = pygame.Rect(track.x, track.y + int((track.height - thumb_h) * ratio), track.width, thumb_h)
            pygame.draw.rect(self.screen, PANEL_ACCENT, thumb, border_radius=3)


    def _draw_panel(self, status):
        layout = self.panel_layout
        pygame.draw.rect(self.screen, PANEL_BG, layout.panel)
        pygame.draw.line(self.screen, PANEL_BORDER, (BOARD_SIZE, 0), (BOARD_SIZE, HEIGHT), 2)

        self._draw_panel_header(status)

        if status.result_text:
            self.screen.blit(self.small_font.render(status.result_text, True, SUCCESS),
                             (layout.notice.x, layout.notice.y))

        self._draw_eval_card(layout.stockfish, "Stockfish eval", status.evals.stockfish, WARNING)
        self._draw_eval_card(layout.neural, "NN eval", status.evals.neural, PANEL_ACCENT)
        self._draw_game_card(status)
        self._draw_engine_card(status)

    def _draw_panel_header(self, status):
        hdr = self.panel_layout.header
        x = hdr.x
        self.screen.blit(self.font.render(status.state_label, True, TEXT_PRIMARY), (x, hdr.y))
        self.screen.blit(self.small_font.render(status.turn_label, True, TEXT_MUTED),
                         (x, hdr.y + max(28, hdr.height // 2 - 2)))
        mode = self.small_font.render(f"Mode {status.mode_label}", True, PANEL_ACCENT)
        self.screen.blit(mode, (x, hdr.bottom - mode.get_height()))

    def _draw_game_card(self, status):
        rect = self.panel_layout.game
        self._draw_card(rect, "Game")
        pad = max(12, rect.width // 20)
        self.screen.blit(self.small_font.render(f"White  {status.players.white}", True, TEXT_PRIMARY),
                         (rect.x + pad, rect.y + max(26, rect.height // 2 - 4)))
        self.screen.blit(self.small_font.render(f"Black  {status.players.black}", True, TEXT_PRIMARY),
                         (rect.x + pad, rect.bottom - max(18, rect.height // 4)))

    def _draw_engine_card(self, status):
        rect = self.panel_layout.engine
        self._draw_card(rect, "Engine")
        pad = max(12, rect.width // 20)
        color = WARNING if status.ai.thinking else SUCCESS
        label = self.small_font.render(status.ai.label, True, color)
        self.screen.blit(label, (rect.x + pad, rect.y + max(26, rect.height // 2 - label.get_height())))
        controls = self.small_font.render("R reset   F flip   U undo   M mode", True, TEXT_MUTED)
        self.screen.blit(controls, (rect.x + pad, rect.bottom - max(16, rect.height // 5) - controls.get_height()))


    def _promotion_layout(self) -> tuple[pygame.Rect, list[pygame.Rect]]:
        menu = pygame.Rect(BOARD_SIZE // 2 - 170, BOARD_SIZE // 2 - 70, 340, 140)
        options = [pygame.Rect(menu.x + 18 + i * 76, menu.y + 48, 64, 64) for i in range(4)]
        return menu, options

    def promotion_piece_from_mouse(self, pos: tuple[int, int]) -> int | None:
        _, option_rects = self._promotion_layout()
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for i, rect in enumerate(option_rects):
            if rect.collidepoint(pos):
                return pieces[i]
        return None

    def _draw_promotion_menu(self, turn: bool):
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        overlay.fill(OVERLAY)
        self.screen.blit(overlay, (0, 0))

        menu_rect, option_rects = self._promotion_layout()
        pygame.draw.rect(self.screen, PANEL_BG, menu_rect, border_radius=14)
        pygame.draw.rect(self.screen, PANEL_BORDER, menu_rect, width=2, border_radius=14)
        self.screen.blit(self.font.render("Choose promotion", True, TEXT_PRIMARY),
                         (menu_rect.x + 18, menu_rect.y + 14))

        symbols = 'QRBN' if turn else 'qrbn'
        for rect, sym in zip(option_rects, symbols):
            pygame.draw.rect(self.screen, GRAY, rect, border_radius=10)
            icon_size = min(rect.width, rect.height) - 8
            icon = pygame.transform.smoothscale(self.pieces[sym], (icon_size, icon_size))
            self.screen.blit(icon, icon.get_rect(center=rect.center))


    def draw(self, game, ui_state, status):
        self.screen.fill(BLACK)

        self._draw_board(game, ui_state)
        self._draw_panel(status)
        self._draw_moves_list(status, ui_state.move_scroll_offset)
        self._draw_eval_bar(status)

        if ui_state.pending_promotion:
            self._draw_promotion_menu(game.board.turn)

        pygame.display.flip()
