# Dataset
FEN = "fen"
LICHESS = "lichess"
DATASET_TYPE = FEN

FEN_PATH = "training_data/fenDataset.csv"
LICHESS_PATH = "training_data/lichessDataset.csv"

MAX_ROWS_LICHESS = 200_000
MAX_ROWS_FEN = 10_000_000

# FEN dataset options
FEN_INCLUDE_MATES = False
FEN_MATE_VALUE_CP = 1000.0
FEN_MAX_VALUE_EVAL = 1000.0


# Training
BATCH_SIZE = 256
EPOCHS = 25
LR = 5e-4
LR_MIN = 1e-5
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 14
SEED = 226
VAL_SPLIT = 20


# Save
SAVE_PATH = f"saved_ai/best_ai_{DATASET_TYPE}.pt"
RESUME_PATH = None