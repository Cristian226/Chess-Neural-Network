import csv
import os
import chess

from ai.encoding import encode_board
from ai.fen_dataset import parse_eval
from ai.preprocess_utils import clear_old_shards, flush_split_buffer
from ai.training_config import *

SPLITS = ("train", "val")
PROGRESS_EVERY = 100_000


def main() -> None:
    os.makedirs(FEN_PREPROCESSED_DIR, exist_ok=True)
    clear_old_shards(FEN_PREPROCESSED_DIR)

    train_cutoff = int(MAX_ROWS_FEN * (1.0 - VAL_SPLIT / 100.0))
    shard_count = dict.fromkeys(SPLITS, 0)
    buffers = {s: {"features": [], "targets": []} for s in SPLITS}
    positions_total = 0
    positions_by_split = dict.fromkeys(SPLITS, 0)

    with open(FEN_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for global_seen, row in enumerate(reader):
            if MAX_ROWS_FEN and global_seen >= MAX_ROWS_FEN:
                break

            split = "train" if global_seen < train_cutoff else "val"

            fen = row.get("FEN", "").strip()
            eval_str = row.get("Evaluation", "").strip()
            if not fen or not eval_str:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            eval_cp = parse_eval(eval_str, board)
            if eval_cp is None:
                continue
            if FEN_MAX_VALUE_EVAL is not None and abs(eval_cp) > FEN_MAX_VALUE_EVAL:
                continue

            buf = buffers[split]
            buf["features"].append(encode_board(board))
            buf["targets"].append(float(eval_cp))
            positions_total += 1
            positions_by_split[split] += 1

            if len(buf["features"]) >= FEN_SHARD_SIZE:
                flush_split_buffer(buffers, split, FEN_PREPROCESSED_DIR, shard_count)

            if (global_seen + 1) % PROGRESS_EVERY == 0:
                print(f"Processed {global_seen + 1:,} rows | positions: {positions_total:,}")

    for split in SPLITS:
        flush_split_buffer(buffers, split, FEN_PREPROCESSED_DIR, shard_count)

    print("FEN preprocessing complete.")
    print(f"Total positions: {positions_total:,}")
    print(f"Train positions: {positions_by_split['train']:,}")
    print(f"Val positions: {positions_by_split['val']:,}")


if __name__ == "__main__":
    main()