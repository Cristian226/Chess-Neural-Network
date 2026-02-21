import os
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ai.dataset import LichessCsvDataset
from ai.fen_dataset import FenDataset
from ai.model import ChessEvalNet
from ai.training_config import *


CLIP_CP = 1000.0

def scale_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.clamp(targets, -CLIP_CP, CLIP_CP) / CLIP_CP

def unscale(preds: torch.Tensor) -> torch.Tensor:
    return preds * CLIP_CP

def train_epoch(model: ChessEvalNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_abs = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(-1)  # [batch_size] - [batch_size, 1]
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
    return avg_loss, avg_abs


if __name__ == "__main__":
    if DATASET_TYPE == "fen":
        dataset = FenDataset()
    else:
        dataset = LichessCsvDataset()
    
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.startswith("cuda"),
        "persistent_workers": NUM_WORKERS > 0,
        "prefetch_factor": 4 if NUM_WORKERS > 0 else 2,
    }
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(DEVICE)
    model = ChessEvalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    best_loss = float('inf')

    os.makedirs("checkpoints", exist_ok=True)
    print(f"Starting training")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        avg_loss, avg_abs = train_epoch(model, loader, optimizer, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f" - Epoch {epoch}: loss={avg_loss:.6f} mae_cp={avg_abs:.2f} lr={current_lr:.2e}")
        
        scheduler.step(avg_loss)
        
        checkpoint_path = os.path.join("checkpoints", f"{DATASET_TYPE}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "clip_cp": CLIP_CP, "epoch": epoch}, checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "clip_cp": CLIP_CP}, SAVE_PATH)

        epoch_time = time.time() - start_time
        print(f" - Epoch time: {epoch_time/60:.2f} minutes")

