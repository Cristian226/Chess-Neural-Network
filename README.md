# Chess AI

A chess program with a graphical board and three move engines: a plain
alpha-beta minimax, a neural network position evaluator searched with alpha-beta,
and Stockfish. It was built as a bachelor's thesis project.

The neural network (ChessEvalNet) is a small residual convolutional network with
squeeze-and-excitation blocks. It reads a board position and returns a single
evaluation score, which the search uses in place of a hand written evaluation
function.

## Requirements

- Python 3.12
- The packages listed in requirements.txt (chess, pygame, torch)
- Stockfish, only if you want to play against it or run the benchmark. It must be
  installed separately and reachable on the system PATH, or you can set its full
  path in config.py.

Install the Python packages with:

    pip install -r requirements.txt

## Running the game

    python main.py

This opens the board window. By default the game runs in player versus engine
mode with you playing White.

## Configuration

Game and engine settings live in config.py. The main options are:

- GAME_MODE: pvp (two humans), pve (human against an engine), or ai_vs_ai
  (two engines play each other).
- HUMAN_COLOR: which side you play in pve mode.
- AI_WHITE_ENGINE and AI_BLACK_ENGINE: which engine each side uses. The choices
  are minimax, neural, and stockfish.
- AI_SEARCH_TIME_LIMIT_MS: thinking time per move for the search engines.
- STOCKFISH_PATH, STOCKFISH_ELO, STOCKFISH_MOVE_TIME_LIMIT: Stockfish settings.
- AI_MODEL_PATH: the trained network used by the neural engine.

Training settings (datasets, batch size, epochs, learning rate, save paths) are
kept separately in ai/training_config.py.

## Project layout

- main.py: entry point that starts the GUI.
- config.py: game and engine configuration.
- gui/: the Pygame interface (board drawing, side panel, input handling).
- core/: game flow, move handling, and the background worker that runs the
  engine without freezing the window.
- engine/: the three engines and the selector that builds one by name.
  - simpleAlphaBetaEngine.py: plain minimax with alpha-beta pruning.
  - nn_engine.py: alpha-beta search using the neural network for evaluation,
    with a transposition table, quiescence search, and the usual move ordering
    and pruning heuristics.
  - stockfish_engine.py: a wrapper around the Stockfish process.
- ai/: the network, board encoding, datasets, training, and tooling.
  - model.py: the ChessEvalNet network.
  - encoding.py: turns a board into the input tensor the network expects.
  - fen_dataset.py, lichess_dataset.py: dataset readers.
  - preprocess_fen.py, preprocess_lichess.py: convert raw data into preprocessed
    shards for faster training.
  - train.py: the training loop.
  - ai_vs_stockfish.py: plays a set of games between a trained network and
    Stockfish and writes the results to CSV.
- assets/: piece images.
- ai_models/: trained network files.

## Training a network

Training uses two datasets, both downloaded from Kaggle:

- A set of positions with engine evaluations, used as the FEN dataset:
  https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
  The reader expects a CSV with FEN and Evaluation columns.
- A set of Lichess games with eval annotations in the movetext, used as the
  Lichess dataset:
  https://www.kaggle.com/datasets/arevel/chess-games

The training mode is set by DATASET_TYPE in ai/training_config.py: fen, lichess,
or combined. The combined mode uses both datasets, so you need both files for it.
Place them where the config points (by default training_data/fenDataset.csv and
training_data/lichessDataset.csv) or change FEN_PATH and LICHESS_PATH to match
where you saved them.

The datasets are large, so there are preprocessing scripts that convert the raw
CSV files into shards that train much faster. Run them once before training:

    python -m ai.preprocess_fen
    python -m ai.preprocess_lichess

Then set the dataset and training options in ai/training_config.py and run:

    python -m ai.train

The script saves the best network to the path given by SAVE_PATH. Point
AI_MODEL_PATH in config.py at a trained file to use it in the game.

## Benchmarking against Stockfish

ai/ai_vs_stockfish.py plays a number of games between one or more trained
networks and Stockfish at a chosen rating, then records each game and a summary.
Edit the engine list and settings at the top of the file, then run:

    python -m ai.ai_vs_stockfish

Stockfish must be available on the PATH for this to work.
