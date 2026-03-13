from dataclasses import dataclass
import pygame

from config import BOARD_SIZE, HEIGHT, PANEL_WIDTH
from core.game_status import GameStatus
from gui.theme import *


MARGIN = 10
GAP = 10
FOOTER_HEIGHT = 28
CARD_PADDING = 10
CARD_RADIUS = 15
ROW_HEIGHT = SMALL_FONT_SIZE + 8
SCROLLBAR_WIDTH = 6
INFO_CARD_RATIO = 0.30
EVAL_CARD_RATIO = 0.17
MOVE_WHITE_OFFSET = 0.25
MOVE_BLACK_OFFSET = 0.65


@dataclass()
class PanelLayout:
    info: pygame.Rect
    stockfish_eval: pygame.Rect
    neural_eval: pygame.Rect
    moves: pygame.Rect
    footer: pygame.Rect
    moves_viewport: pygame.Rect
    visible_move_rows: int


class PanelRenderer:
    def __init__(self):
        pygame.font.init()
        self.font_small = pygame.font.SysFont(FONT_NAME, SMALL_FONT_SIZE)
        self.font_medium = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        self.font_large = pygame.font.SysFont(FONT_NAME, LARGE_FONT_SIZE)
        self.layout = self._build_layout()

    def draw(self, screen, status, gui_state):
        panel = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, panel)
        pygame.draw.line(screen, PANEL_BORDER_COLOR, (BOARD_SIZE, 0), (BOARD_SIZE, HEIGHT), 2)

        self._draw_info_card(screen, status)
        self._draw_eval_card(screen, self.layout.stockfish_eval, "Stockfish eval", status.stockfish_eval)
        self._draw_eval_card(screen, self.layout.neural_eval, "NN eval", status.neural_eval)
        self._draw_moves_card(screen, status, gui_state)
        self._draw_footer(screen)


    def _build_layout(self):
        panel_x = BOARD_SIZE + MARGIN
        panel_h = HEIGHT - 2 * MARGIN
        panel_right = BOARD_SIZE + PANEL_WIDTH - MARGIN
        card_width = panel_right - panel_x
        usable_h = panel_h - FOOTER_HEIGHT - 3 * GAP
        info_h = int(usable_h * INFO_CARD_RATIO)
        eval_h = int(usable_h * EVAL_CARD_RATIO)

        y = MARGIN
        info = pygame.Rect(panel_x, y, card_width, info_h); y += info_h + GAP
        stockfish = pygame.Rect(panel_x, y, card_width, eval_h); y += eval_h + GAP
        neural = pygame.Rect(panel_x, y, card_width, eval_h); y += eval_h + GAP
        footer = pygame.Rect(panel_x + MARGIN, panel_h - FOOTER_HEIGHT + MARGIN, card_width, FOOTER_HEIGHT)

        moves_h = footer.y - y - GAP
        moves = pygame.Rect(panel_x, y, card_width, moves_h)
        viewport = pygame.Rect( moves.x + CARD_PADDING,
            moves.y + self.font_medium.get_height() + 12,
            moves.width - 2 * CARD_PADDING - SCROLLBAR_WIDTH - 6,
            moves.height - self.font_medium.get_height() - 20)

        visible_rows = max(1, viewport.height // ROW_HEIGHT)

        return PanelLayout(info, stockfish, neural, moves, footer, viewport, visible_rows)

    def _draw_card(self, screen, rect):
        pygame.draw.rect(screen, PANEL_BG, rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, 2, border_radius=CARD_RADIUS)
        return rect.x + CARD_PADDING, rect.y + CARD_PADDING

    def _draw_info_card(self, screen, status: GameStatus):
        rect = self.layout.info
        x, y = self._draw_card(screen, rect)

        mode = status.mode_label.replace("_", " ").upper()
        text = self.font_large.render(mode, True, PANEL_ACCENT)
        screen.blit(text, (x, y))
        y += text.get_height()

        turn = "WHITE" if status.turn_label.lower().startswith("white") else "BLACK"
        text = self.font_medium.render(f"{turn} to move - {status.state_label}", True, PANEL_ACCENT)
        screen.blit(text, (x, y))
        y += text.get_height() + 4

        text = self.font_small.render(f"White: {status.white_player}", True, TEXT_COLOR)
        screen.blit(text, (x, y))
        y += text.get_height()

        text = self.font_small.render(f"Black: {status.black_player}", True, TEXT_COLOR)
        screen.blit(text, (x, y))
        y += text.get_height() + 4

        color = WARNING_COLOR if status.ai_thinking else TEXT_MUTED
        text = self.font_small.render(status.ai_label, True, color)
        screen.blit(text, (x, y))

    def _draw_eval_card(self, screen, rect, title, eval_data):
        x, y = self._draw_card(screen, rect)

        text = self.font_medium.render(title, True, PANEL_ACCENT)
        screen.blit(text, (x, y))
        y += text.get_height() + 2

        label = f"{eval_data.label}…" if eval_data.pending else eval_data.label
        color = WARNING_COLOR if eval_data.pending else TEXT_COLOR
        text = self.font_medium.render(label, True, color)
        screen.blit(text, (x, y))

        bar = pygame.Rect(x, rect.bottom - 22, rect.width - 2 * CARD_PADDING, 8)
        pygame.draw.rect(screen, EVAL_BAR_BG, bar, border_radius=4)

        fill = int(bar.width * max(0, min(1, eval_data.fill)))
        if fill > 0:
            pygame.draw.rect(screen, TEXT_COLOR, (bar.x, bar.y, fill, bar.height), border_radius=4)

        if eval_data.error:
            err = self.font_small.render("eval unavailable", True, CHECK_ALERT_COLOR)
            screen.blit(err, (rect.right - err.get_width() - CARD_PADDING, rect.y + 8))

    def _draw_moves_card(self, screen, status, gui_state):
        rect = self.layout.moves
        viewport = self.layout.moves_viewport

        x, y = self._draw_card(screen, rect)

        title = self.font_medium.render("Moves", True, PANEL_ACCENT)
        screen.blit(title, (x, y))

        pygame.draw.rect(screen, MOVE_VIEWPORT_BG, viewport, border_radius=8)
        rows = status.move_rows
        visible = self.layout.visible_move_rows

        max_start = max(0, len(rows) - visible)
        start = min(gui_state.move_scroll_offset, max_start)

        num_x = viewport.x + 6
        white_x = viewport.x + int(viewport.width * MOVE_WHITE_OFFSET)
        black_x = viewport.x + int(viewport.width * MOVE_BLACK_OFFSET)

        prev_clip = screen.get_clip()
        screen.set_clip(viewport)
        y = viewport.y + 4
        for i, (num, white, black) in enumerate(rows[start:start + visible]):
            if i % 2 == 0:
                pygame.draw.rect(screen, MOVES_ROW_ALT_COLOR, (viewport.x + 2, y, viewport.width - 4, ROW_HEIGHT), border_radius=5)

            screen.blit(self.font_small.render(num, True, TEXT_MUTED), (num_x, y))
            screen.blit(self.font_small.render(white, True, TEXT_COLOR), (white_x, y))
            screen.blit(self.font_small.render(black, True, TEXT_COLOR), (black_x, y))

            y += ROW_HEIGHT
        screen.set_clip(prev_clip)

        if len(rows) > visible:
            track = pygame.Rect(rect.right - SCROLLBAR_WIDTH - 6, viewport.y, SCROLLBAR_WIDTH, viewport.height)
            pygame.draw.rect(screen, EVAL_BAR_BG, track, border_radius=3)

            thumb_h = max(20, int(track.height * visible / len(rows)))
            ratio = start / max(1, max_start)
            thumb_y = track.y + int((track.height - thumb_h) * ratio)

            pygame.draw.rect(screen, PANEL_ACCENT, (track.x, thumb_y, track.width, thumb_h), border_radius=3)

    def _draw_footer(self, screen):
        footer = self.layout.footer
        text = "R reset   F flip   U undo   M mode"
        text = self.font_small.render(text, True, TEXT_MUTED)
        y = footer.y + (footer.height - text.get_height()) // 2
        screen.blit(text, (footer.x, y))
