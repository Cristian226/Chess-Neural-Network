from core.game_manager import GameManager
from gui.game_gui import GameGUI

if __name__ == "__main__":
    manager = GameManager()
    gui = GameGUI(manager)
    gui.run()
