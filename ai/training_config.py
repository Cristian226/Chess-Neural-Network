# Dataset
FEN = "fen"
LICHESS = "lichess"
DATASET_TYPE = LICHESS

FEN_PATH = "training_data/fenDataset.csv"
LICHESS_PATH = "training_data/lichessDataset.csv"

MAX_ROWS_LICHESS = 500_000
MAX_ROWS_FEN = 13_000_000

# Preprocessed Lichess options
USE_PREPROCESSED_LICHESS = True
LICHESS_PREPROCESSED_DIR = r"D:\Workplace\Licenta\lichess_preprocessed"
LICHESS_SHARD_SIZE = 10_000

# Preprocessed FEN options
USE_PREPROCESSED_FEN = True
FEN_PREPROCESSED_DIR = r"D:\Workplace\Licenta\fen_preprocessed"
FEN_SHARD_SIZE = 10_000

# FEN dataset options
FEN_INCLUDE_MATES = False
FEN_MATE_VALUE_CP = 1000.0
FEN_MAX_VALUE_EVAL = None


# Training
BATCH_SIZE = 512
EPOCHS = 35
LR = 5e-4
LR_MIN = 1e-4
WEIGHT_DECAY = 8e-4
NUM_WORKERS = 12
SEED = 226
VAL_SPLIT = 20
TANH_SCALE = 600.0


# Save
SAVE_PATH = f"saved_ai/best_ai_{DATASET_TYPE}.pt"
RESUME_PATH = None