import os
import time
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ai.lichess_dataset import LichessCsvDataset
from ai.fen_dataset import FenDataset
from ai.model import ChessEvalNet
from ai.training_config import *

DEVICE = torch.device("cuda")
CLIP_CP = 1000.0

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loader_kwargs():
    return dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

def scale_targets(t: torch.Tensor):
    return torch.clamp(t, -CLIP_CP, CLIP_CP) / CLIP_CP

def unscale(t: torch.Tensor):
    return t * CLIP_CP


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.to(DEVICE)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, best_loss


def run_epoch(model, loader, optimizer=None, scaler=None) -> Tuple[float, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_abs = 0.0
    total_count = 0

    for features, targets in loader:
        features = features.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True).unsqueeze(1)

        scaled_targets = scale_targets(targets)
        with torch.autocast(device_type="cuda", enabled=True):
            preds = model(features)
            loss = F.smooth_l1_loss(preds, scaled_targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_abs += torch.mean(torch.abs(unscale(preds.detach()) - targets)).item() * batch_size
        total_count += batch_size

    return total_loss / total_count, total_abs / total_count


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

    model = ChessEvalNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_val_loss = float("inf")
    start_epoch = 1

    os.makedirs("checkpoints", exist_ok=True)

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        start_epoch, best_val_loss = load_checkpoint(RESUME_PATH, model, optimizer)
    elif RESUME_PATH:
        print("Checkpoint not found:", RESUME_PATH)

    print(f"Starting training on {DATASET_TYPE}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_mae = run_epoch(model, val_loader)

        print(f"Epoch {epoch}")
        print(f"Train loss={train_loss:.6f} mae={train_mae:.1f}cp")
        print(f"Val loss={val_loss:.6f} mae={val_mae:.1f}cp")

        save_checkpoint(f"checkpoints/{DATASET_TYPE}_epoch{epoch}.pt", model, optimizer, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(SAVE_PATH, model, optimizer, epoch, best_val_loss)

        print(f"Epoch time: {(time.time() - t0)/60:.2f} min")