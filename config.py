import chess

# Game Modes
PVP = "pvp"
PVE = "pve"
AI_VS_AI = "ai_vs_ai"

GAME_MODE = PVE
HUMAN_COLOR = chess.WHITE   # used in PVE mode
AI_COLOR = not HUMAN_COLOR  # used in PVE mode
AI_SAVE_PGN = False


# Engine selection
DEFAULT_MINMAX_AI = "minimax"
NEURAL_NETWORK_AI = "neural"
STOCKFISH_AI = "stockfish"

AI_WHITE_ENGINE = STOCKFISH_AI
AI_BLACK_ENGINE = NEURAL_NETWORK_AI
AI_MINMAX_DEPTH = 7
AI_SEARCH_TIME_LIMIT_MS = 2_500    # ms per move


# GUI layout
DRAG_DISTANCE_THRESHOLD_SQ = 100
BOARD_SIZE = 640
PANEL_WIDTH = 360
WIDTH = BOARD_SIZE + PANEL_WIDTH
HEIGHT = BOARD_SIZE
SQUARE_SIZE = BOARD_SIZE // 8


# Stockfish settings
STOCKFISH_PATH = None               # None = use system PATH
STOCKFISH_MOVE_TIME_LIMIT = 1       # seconds per move
STOCKFISH_ELO = 2500                # 1320 – 3190
STOCKFISH_EVAL_TIME_LIMIT = 0.5
STOCKFISH_EVAL_ELO = None           # None = full strength


# Paths
PIECE_ASSET_PATH = "assets/pieces/"
AI_MODEL_PATH = r"ai_models\ai_combined.pt"
AI_PGN_PATH = "logs/ai_games.pgn"