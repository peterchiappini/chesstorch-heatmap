"""
Training loop for ChessCNN.

Usage:
    python train.py magnus --epochs 10
    python train.py hikaru --epochs 10
    python train.py --pgn custom.pgn --weights custom.pth
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from board_encoder import load_or_parse_pgn
from model import ChessCNN, get_device


def train_model(
    pgn_path: str,
    weights_path: str = "chess_model.pth",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> ChessCNN:
    """Train ChessCNN on positions extracted from a PGN file.

    Args:
        pgn_path:   Path to a .pgn file (may contain multiple games).
        epochs:     Number of full passes over the dataset.
        batch_size: Mini-batch size for the DataLoader.
        lr:         Learning rate for Adam.

    Returns:
        The trained model (still on its device).
    """
    device = get_device()
    print(f"Using device: {device}")

    # ── Load tensors (parses + caches if needed) ──
    tensors, from_squares, to_squares, num_games = load_or_parse_pgn(pgn_path)

    dataset = TensorDataset(tensors, from_squares, to_squares)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Model, loss, optimizer ──
    model = ChessCNN().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ──
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_x, target_from, target_to in loader:
            batch_x = batch_x.to(device)
            target_from = target_from.to(device)
            target_to = target_to.to(device)

            optimizer.zero_grad()
            logits_from, logits_to = model(batch_x)
            loss_from = criterion(logits_from, target_from)
            loss_to = criterion(logits_to, target_to)
            loss = loss_from + loss_to
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            correct += (logits_to.argmax(dim=1) == target_to).sum().item()
            total += batch_x.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"acc={accuracy:.2%}  "
            f"({elapsed:.1f}s)"
        )

    # ── Save trained weights ──
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # ── Save training metadata ──
    meta = {
        "games": num_games,
        "epochs": epochs,
        "final_loss": round(avg_loss, 4),
        "final_accuracy": round(accuracy, 4),
        "positions": len(tensors),
        "batch_size": batch_size,
        "lr": lr,
    }
    meta_path = weights_path.replace(".pth", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training metadata saved to {meta_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChessCNN on a PGN file")
    parser.add_argument("player", nargs="?", default=None, help="Player name (e.g. magnus, hikaru, eric)")
    parser.add_argument("--pgn", type=str, default=None, help="Custom PGN path (overrides player)")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights output path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.pgn:
        pgn_path = args.pgn
        w_path = args.weights or "chess_model.pth"
    elif args.player:
        pgn_path = os.path.join("data", f"{args.player}.pgn")
        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        w_path = os.path.join(weights_dir, f"{args.player}.pth")
    else:
        pgn_path = "test.pgn"
        w_path = "chess_model.pth"

    model = train_model(pgn_path, weights_path=w_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print("Training complete.")
