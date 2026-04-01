import csv
import os
import random

from ai.lichess_dataset import parse_movetext
from ai.preprocess_utils import clear_old_shards, flush_split_buffer
from ai.training_config import *

PROGRESS_EVERY = 10_000
SPLITS = ("train", "val")


def main() -> None:
    random.seed(SEED)
    os.makedirs(LICHESS_PREPROCESSED_DIR, exist_ok=True)
    clear_old_shards(LICHESS_PREPROCESSED_DIR)

    train_cutoff = int(MAX_ROWS_LICHESS * (1.0 - VAL_SPLIT / 100.0))
    shard_count = dict.fromkeys(SPLITS, 0)
    buffers = {s: {"features": [], "targets": []} for s in SPLITS}
    positions_total = 0
    positions_by_split = dict.fromkeys(SPLITS, 0)

    with open(LICHESS_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for global_seen, row in enumerate(reader):
            if MAX_ROWS_LICHESS and global_seen >= MAX_ROWS_LICHESS:
                break

            split = "train" if global_seen < train_cutoff else "val"

            samples = parse_movetext(row.get("AN", ""))
            if not samples:
                continue

            buf = buffers[split]
            for features, eval_cp in samples:
                buf["features"].append(features)
                buf["targets"].append(float(eval_cp))
                positions_total += 1
                positions_by_split[split] += 1

                if len(buf["features"]) >= LICHESS_SHARD_SIZE:
                    flush_split_buffer(buffers, split, LICHESS_PREPROCESSED_DIR, shard_count)

            if (global_seen + 1) % PROGRESS_EVERY == 0:
                print(f"Processed {global_seen + 1:,} games | positions: {positions_total:,}")

    for split in SPLITS:
        flush_split_buffer(buffers, split, LICHESS_PREPROCESSED_DIR, shard_count)

    print("Preprocessing complete.")
    print(f"Total positions: {positions_total:,}")
    print(f"Train positions: {positions_by_split['train']:,}")
    print(f"Val positions: {positions_by_split['val']:,}")


if __name__ == "__main__":
    main()