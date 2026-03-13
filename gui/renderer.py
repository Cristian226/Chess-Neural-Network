import pygame

from config import *
from gui.board_renderer import BoardRenderer
from gui.panel_renderer import PanelRenderer


class Renderer:
    def __init__(self):
        pygame.font.init()
        self._load_assets()
        self.board_renderer = BoardRenderer(self.pieces)
        self.panel_renderer = PanelRenderer()
        self.panel_layout = self.panel_renderer.layout

    def _load_assets(self):
        self.pieces = {}
        for symbol in "PNBRQKpnbrqk":
            name = ("w" if symbol.isupper() else "b") + symbol.lower()
            image = pygame.image.load(f"{PIECE_ASSET_PATH}{name}.png").convert_alpha()
            self.pieces[symbol] = pygame.transform.smoothscale(image, (SQUARE_SIZE, SQUARE_SIZE))

    def draw(self, screen, manager, gui_state, status):
        screen.fill((0, 0, 0))
        self.board_renderer.draw(screen, manager.board, gui_state)
        self.panel_renderer.draw(screen, status, gui_state)
        pygame.display.flip()
