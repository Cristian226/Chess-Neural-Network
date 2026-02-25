import os
import time
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ai.lichess_dataset import LichessCsvDataset
from ai.fen_dataset import FenDataset
from ai.model import ChessEvalNet
from ai.training_config import *

CLIP_CP = 1000.0
DEVICE = torch.device("cuda")


def build_loader_kwargs() -> dict:
    kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    return kwargs

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def scale_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.clamp(targets, -CLIP_CP, CLIP_CP) / CLIP_CP

def unscale(preds: torch.Tensor) -> torch.Tensor:
    return preds * CLIP_CP


def build_checkpoint(model: ChessEvalNet, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }

def load_checkpoint(path: str, model: ChessEvalNet, optimizer: torch.optim.Optimizer) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)

    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))

    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, best_loss


def train_epoch(model: ChessEvalNet, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_abs = 0.0

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(-1) # [batch_size] - [batch_size, 1]
        scaled_targets = scale_targets(targets)

        preds = model(features)
        loss = torch.nn.functional.smooth_l1_loss(preds, scaled_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_abs += torch.mean(torch.abs(unscale(preds.detach()) - targets)).item() * batch_size

    avg_loss = total_loss / total_count
    avg_abs = total_abs / total_count
    print(f" - Train: loss={avg_loss:.6f} mae_cp={avg_abs:.2f}")
    return avg_loss, avg_abs

@torch.no_grad()
def eval_epoch(model: ChessEvalNet, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_abs = 0.0

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(-1)
        scaled_targets = scale_targets(targets)

        preds = model(features)
        loss = torch.nn.functional.smooth_l1_loss(preds, scaled_targets)

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_abs += torch.mean(torch.abs(unscale(preds) - targets)).item() * batch_size

    avg_loss = total_loss / total_count
    avg_abs = total_abs / total_count
    print(f" - Val: loss={avg_loss:.6f} mae_cp={avg_abs:.2f}")
    return avg_loss, avg_abs


if __name__ == "__main__":
    set_seed(SEED)

    if DATASET_TYPE == FEN:
        train_dataset = FenDataset(split="train")
        val_dataset = FenDataset(split="val")
    else:
        train_dataset = LichessCsvDataset(split="train")
        val_dataset = LichessCsvDataset(split="val")

    loader_kwargs = build_loader_kwargs()
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    model = ChessEvalNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val_loss = float("inf")
    start_epoch = 1

    os.makedirs("checkpoints", exist_ok=True)

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        start_epoch, best_val_loss = load_checkpoint(RESUME_PATH, model, optimizer)
    else:
        if RESUME_PATH:
            print(f"Resume checkpoint not found: {RESUME_PATH}")
        model.to(DEVICE)

    print("Starting training")

    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_abs = train_epoch(model, train_loader, optimizer)
        val_loss, val_abs = eval_epoch(model, val_loader)

        checkpoint = build_checkpoint(model, optimizer, epoch, best_val_loss)
        torch.save(checkpoint, f"checkpoints/{DATASET_TYPE}_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_loss"] = best_val_loss
            torch.save(checkpoint, SAVE_PATH)

        epoch_time = time.time() - start_time
        print(f" - Epoch time: {epoch_time/60:.2f} minutes")

