import os
import time
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ai.lichess_dataset import LichessCsvDataset
from ai.preprocessed_dataset import PreprocessedShardDataset
from ai.fen_dataset import FenDataset
from ai.model import ChessEvalNet, CHANNELS, NUM_BLOCKS
from ai.training_config import *

DEVICE = torch.device("cuda")

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loader_kwargs():
    return dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

def scale_targets(t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(t / TANH_SCALE)   # cp → (-1, +1)

def unscale(t: torch.Tensor) -> torch.Tensor:
    return torch.atanh(t.float().clamp(-1 + 1e-4, 1 - 1e-4)) * TANH_SCALE  # (-1, +1) → cp


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "tanh_scale": TANH_SCALE,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
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
        with torch.autocast("cuda"):
            preds = model(features)
            loss = F.smooth_l1_loss(preds, scaled_targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_abs += torch.mean(torch.abs(unscale(preds.detach()) - targets)).item() * batch_size
        total_count += batch_size

    return total_loss / total_count, total_abs / total_count


def build_datasets():
    if DATASET_TYPE == COMBINED:
        train_dataset = PreprocessedShardDataset(split="train", data_dirs=[FEN_PREPROCESSED_DIR, LICHESS_PREPROCESSED_DIR])
        val_dataset = PreprocessedShardDataset(split="val", data_dirs=[FEN_PREPROCESSED_DIR, LICHESS_PREPROCESSED_DIR])
    elif DATASET_TYPE == FEN:
        if USE_PREPROCESSED_FEN:
            train_dataset = PreprocessedShardDataset(split="train", data_dirs=[FEN_PREPROCESSED_DIR])
            val_dataset = PreprocessedShardDataset(split="val", data_dirs=[FEN_PREPROCESSED_DIR])
        else:
            train_dataset = FenDataset(split="train")
            val_dataset = FenDataset(split="val")
    else:
        if USE_PREPROCESSED_LICHESS:
            train_dataset = PreprocessedShardDataset(split="train", data_dirs=[LICHESS_PREPROCESSED_DIR])
            val_dataset = PreprocessedShardDataset(split="val", data_dirs=[LICHESS_PREPROCESSED_DIR])
        else:
            train_dataset = LichessCsvDataset(split="train")
            val_dataset = LichessCsvDataset(split="val")

    return train_dataset, val_dataset


if __name__ == "__main__":
    set_seed(SEED)
    train_dataset, val_dataset = build_datasets()
    loader_kwargs = build_loader_kwargs()
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    model = ChessEvalNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_val_loss = float("inf")
    start_epoch = 1

    os.makedirs("checkpoints", exist_ok=True)
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        start_epoch, best_val_loss = load_checkpoint(RESUME_PATH, model, optimizer, scheduler)
    elif RESUME_PATH:
        print("Checkpoint not found:", RESUME_PATH)

    preprocessed_str = "preprocessed " if (USE_PREPROCESSED_LICHESS or USE_PREPROCESSED_FEN) else ""
    print(f"Starting training on {preprocessed_str}{DATASET_TYPE}")
    print(f"Channels: {CHANNELS}, Blocks: {NUM_BLOCKS}, Batch: {BATCH_SIZE}, LR: {LR}, LR min: {LR_MIN}, WeightDecay: {WEIGHT_DECAY}, TanhScale: {TANH_SCALE}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_mae = run_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_mae = run_epoch(model, val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        save_checkpoint(f"checkpoints/{DATASET_TYPE}_epoch{epoch}.pt", model, optimizer, scheduler, epoch, best_val_loss)

        print(f"Epoch {epoch}                 lr ={current_lr:.2e}")
        print(f"Train loss={train_loss:.6f}   mae={train_mae:.1f}cp")
        print(f"Val   loss={val_loss:.6f}   mae={val_mae:.1f}cp   ")
        print(f"Epoch time: {(time.time() - t0)/60:.2f} min")
