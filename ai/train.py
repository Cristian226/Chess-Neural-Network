import argparse
import os
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ai.dataset import LichessCsvDataset
from ai.fen_dataset import FenDataset
from ai.model import ChessEvalNet
from config import *


CLIP_CP = 1000.0

def _scale_targets(targets: torch.Tensor) -> torch.Tensor:
    return torch.clamp(targets, -CLIP_CP, CLIP_CP) / CLIP_CP

def _unscale(preds: torch.Tensor) -> torch.Tensor:
    return preds * CLIP_CP

def train_epoch(
    model: ChessEvalNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_abs = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(-1)  # [batch_size] - [batch_size, 1]
        scaled_targets = _scale_targets(targets)

        preds = model(features)
        loss = torch.nn.functional.smooth_l1_loss(preds, scaled_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        total_abs += torch.mean(torch.abs(_unscale(preds.detach()) - targets)).item() * batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_abs = total_abs / max(total_count, 1)
    return avg_loss, avg_abs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--dataset-type", choices=["lichess", "fen"], default="fen", help="Dataset type: lichess (movetext) or fen (positions)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default=BEST_AI_MODEL_PATH)
    args = parser.parse_args()

    if args.dataset_type == "fen":
        dataset = FenDataset(args.data, args.max_rows)
    else:
        dataset = LichessCsvDataset(args.data, args.max_rows, args.min_elo)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = ChessEvalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float('inf')
    print(f"Starting training")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        avg_loss, avg_abs = train_epoch(model, loader, optimizer, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f" - Epoch {epoch}: loss={avg_loss:.6f} mae_cp={avg_abs:.2f} lr={current_lr:.2e}")
        
        scheduler.step(avg_loss)
        
        checkpoint_path = f"checkpoints/{args.save[:-3]}_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "clip_cp": CLIP_CP, "epoch": epoch}, checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "clip_cp": CLIP_CP}, args.save)

        epoch_time = time.time() - start_time
        print(f" - Epoch time: {epoch_time/60:.2f} minutes")

if __name__ == "__main__":
    main()

