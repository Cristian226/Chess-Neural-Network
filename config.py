import chess

# Game settings
PVP = 'pvp'
PVE = 'pve'
AI_VS_AI = 'ai_vs_ai'

GAME_MODE = PVE  # change to swap between modes
HUMAN_COLOR = chess.WHITE  # color for human player in 'pve' mode
AI_COLOR = not HUMAN_COLOR  # color for human 2 or AI in 'pve' mode
AI_SAVE_PGN = True     # save completed AI vs AI games to PGN when true


# Engine selection: 'minimax', 'neural' (or 'nn'), 'stockfish' and depth
DEFAULT_MIN_MAX_AI = 'minimax'
NEURAL_NETWORK_AI = 'neural'
STOCKFISH_AI = 'stockfish'

AI_WHITE_ENGINE = STOCKFISH_AI
AI_BLACK_ENGINE = NEURAL_NETWORK_AI
AI_MINMAX_DEPTH = 4
AI_SEARCH_TIME_LIMIT_MS = 10000


# Stockfish settings
STOCKFISH_PATH = None  # path to stockfish executable (None = use PATH)
STOCKFISH_TIME_LIMIT = 1  # time limit per move in seconds
STOCKFISH_ELO = 1800   # min 1320, max 3190
STOCKFISH_EVAL_TIME_LIMIT = 0.35  # analysis time for the eval bar in seconds
STOCKFISH_EVAL_ELO = None  # None = full-strength eval, or set an Elo to limit it


# GUI settings
BOARD_SIZE = 640
PANEL_WIDTH = 320
WIDTH = BOARD_SIZE + PANEL_WIDTH
HEIGHT = BOARD_SIZE
SQ_SIZE = BOARD_SIZE // 8
FONT_SIZE = 28
SMALL_FONT_SIZE = 20
FPS = 144


# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
LAST_MOVE = (246, 246, 105)
SELECTED = (100, 150, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
PANEL_BG = (24, 28, 36)
CARD_BG = (30, 35, 44)
MOVES_BG = (24, 28, 36)
MOVES_ROW_ALT = (35, 40, 50)
SCROLLBAR_TRACK = (44, 50, 62)
PANEL_BORDER = (51, 58, 72)
PANEL_ACCENT = (70, 120, 255)
TEXT_PRIMARY = (242, 244, 248)
TEXT_MUTED = (170, 178, 190)
SUCCESS = (110, 196, 120)
WARNING = (235, 187, 74)
MOVE_DOT = (62, 92, 158)
CAPTURE_HINT = (194, 88, 72)
CHECK_ALERT = (206, 76, 76)
OVERLAY = (10, 12, 18, 190)


# Paths
PIECE_PATH = "assets/pieces/"
AI_MODEL_PATH = "saved_ai/ai4lichess.pt"  # path to trained PyTorch model
AI_PGN_PATH = "logs/ai_games.pgn"  # appended file for AI game results
