import os
import torch


def _is_shard_file(name: str) -> bool:
    return name.endswith(".pt") and (name.startswith("train_") or name.startswith("val_"))

def save_shard(out_dir: str, split: str, shard_idx: int, features: list[torch.Tensor], targets: list[float]) -> None:
    shard_path = os.path.join(out_dir, f"{split}_{shard_idx:06d}.pt")
    torch.save(
        {
            "features": torch.stack(features).to(torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
        },
        shard_path,
    )

def flush_split_buffer(buffers: dict[str, dict[str, list]], split: str, out_dir: str, shard_count: dict[str, int]) -> None:
    buf = buffers[split]
    if not buf["features"]:
        return
    save_shard(out_dir, split, shard_count[split], **buf)
    shard_count[split] += 1
    buf["features"].clear()
    buf["targets"].clear()

def clear_old_shards(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        return

    for name in os.listdir(out_dir):
        if _is_shard_file(name):
            os.remove(os.path.join(out_dir, name))
