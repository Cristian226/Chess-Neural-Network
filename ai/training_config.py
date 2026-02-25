# Dataset
FEN = "fen"
LICHESS = "lichess"
DATASET_TYPE = FEN

FEN_PATH = "path-to-FEN.csv"
LICHESS_PATH = "path-to-lichess.csv"
DATASET_PATH = FEN_PATH if DATASET_TYPE == FEN else LICHESS_PATH

MAX_ROWS = 10_000_000 # 200_000 for lichess, 10_000_000 for FEN

# FEN dataset options
FEN_INCLUDE_MATES = False
FEN_MATE_VALUE_CP = 1000.0
FEN_MAX_VALUE_EVAL = 2000.0


# Training
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 14
SEED = 42
VAL_SPLIT = 20


# Save
SAVE_PATH = f"saved_ai/best_ai_{DATASET_TYPE}.pt"
RESUME_PATH = "checkpoints\\fen_epoch2.pt"