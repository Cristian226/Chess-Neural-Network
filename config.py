import chess

# Game settings
PVP = 'pvp'
PVE = 'pve'
AI_VS_AI = 'ai_vs_ai'

GAME_MODE = AI_VS_AI  # change to swap between modes
HUMAN_COLOR = chess.WHITE  # color for human player in 'pve' mode
AI_COLOR = HUMAN_COLOR ^ True  # color for human 2 or AI in 'pve' mode
AI_SAVE_PGN = True     # save completed AI vs AI games to PGN when true


# Engine selection: 'minimax', 'neural' (or 'nn'), 'stockfish' and depth
DEFAULT_MIN_MAX = 'minimax'
NEURAL_NETWORK = 'neural'
STOCKFISH = 'stockfish'

AI_WHITE_ENGINE = NEURAL_NETWORK
AI_BLACK_ENGINE = DEFAULT_MIN_MAX
AI_WHITE_DEPTH = 4
AI_BLACK_DEPTH = 4


# Stockfish settings
STOCKFISH_PATH = None  # path to stockfish executable (None = use PATH)
STOCKFISH_TIME_LIMIT = 0.1  # time limit per move in seconds
STOCKFISH_DEPTH = 20

AI_MOVE_DELAY_MS = 100  # delay between AI moves (milliseconds) for smoother play


# GUI settings
WIDTH = 640
HEIGHT = 700
BOARD_SIZE = 640
SQ_SIZE = 80
FONT_SIZE = 32
FPS = 144


# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)  
LAST_MOVE = (246, 246, 105) 
SELECTED = (100, 150, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GRAY = (220, 220, 220)


# Paths
PIECE_PATH = "assets/pieces/"
AI_MODEL_PATH = 'saved_ai/best.pt'  # path to trained PyTorch model
AI_PGN_PATH = "Logs/ai_games.pgn"  # appended file for AI game results
BEST_AI_MODEL_PATH = 'saved_ai/best_ai.pt'  # path to best model based on eval score
