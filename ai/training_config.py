import torch

# Dataset
FEN = "fen"
LICHESS = "lichess"
DATASET_TYPE = FEN
DATASET_PATH = "data/lichess_db_eval.csv"
MAX_ROWS = 10_000_000

# Lichess dataset options
MIN_ELO = None 

# FEN dataset options
FEN_INCLUDE_MATES = False
FEN_MATE_VALUE_CP = 1000.0
FEN_MAX_VALUE_EVAL = 2000.0


# Training
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 16


# Device & save
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = f"saved_ai/best_ai_{DATASET_TYPE}.pt"
