"""
Grad-CAM heatmap generation for ChessCNN.

Converts a FEN string into an 8×8 attention heatmap by running a forward
and backward pass through the model, then combining the final conv layer's
activations and gradients.
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch
import torch.nn.functional as F

from board_encoder import board_to_tensor
from model import ChessCNN

# --- Heatmap diffusion parameters ---
TEMPERATURE = 0.4       # Lower = more spread out (divides raw CAM before norm)
MIN_HEAT_FLOOR = 0.05   # Baseline glow so no square is fully "dead"
GAUSSIAN_SIGMA = 0.6    # Spatial smoothing across neighboring squares


VALID_DEPTHS = ("layer1", "layer2", "layer3", "layer4")


def _sq_to_display(sq: int) -> int:
    """Convert a python-chess square index (a1=0) to display index (a8=0).

    python-chess: row 0 = rank 1 (bottom).  Frontend SQUARES: row 0 = rank 8 (top).
    Flip the rank while keeping the file.
    """
    rank, file = divmod(sq, 8)
    return (7 - rank) * 8 + file


def generate_heatmap(model: ChessCNN, fen: str, depth: str = "layer3") -> dict:
    """Return Grad-CAM heatmap and predicted move for the given board position.

    Args:
        model: A ChessCNN instance (should be in .eval() mode).
        fen: A standard FEN string describing the board position.
        depth: Which conv layer to visualise ("early", "mid", "deep", "final").

    Returns:
        A dict with:
          - "weights": list of 64 floats (0.0–1.0), rank-descending file-ascending.
          - "predicted_move": [from_square_name, to_square_name].

    Raises:
        ValueError: If the FEN string cannot be parsed.
    """
    # --- Parse FEN into a board tensor ---
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {e}") from e

    # --- Early exit for game-over positions ---
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            status = f"Checkmate — {winner} wins"
            result = "0-1" if board.turn == chess.WHITE else "1-0"
        else:
            status = "Stalemate — Draw"
            result = "1/2-1/2"
        return {
            "weights": [0.0] * 64,
            "top_moves": [],
            "game_over": status,
            "result": result,
        }

    tensor = board_to_tensor(board).unsqueeze(0)  # (1, 17, 8, 8)

    # --- Forward pass (with gradients enabled for Grad-CAM) ---
    tensor.requires_grad_(True)
    logits_from, logits_to = model(tensor)  # each (1, 64)

    # --- Score every legal move using both heads ---
    move_scores = []
    for move in legal_moves:
        score = (logits_from[0, move.from_square] + logits_to[0, move.to_square])
        move_scores.append((move, score))

    # Sort by score descending and take top 3
    move_scores.sort(key=lambda x: x[1].item(), reverse=True)
    top_n = move_scores[:3]

    # Compute confidence via softmax over all legal move scores
    all_scores = torch.tensor([s.item() for _, s in move_scores])
    probs = F.softmax(all_scores, dim=0)

    top_moves = []
    for i, (move, _score) in enumerate(top_n):
        top_moves.append({
            "move": [chess.SQUARE_NAMES[move.from_square], chess.SQUARE_NAMES[move.to_square]],
            "confidence": round(probs[i].item(), 4),
        })

    # Use the #1 move for Grad-CAM
    best_move = top_n[0][0]
    from_idx = best_move.from_square
    to_idx = best_move.to_square

    # Clamp depth to a valid layer name
    layer = depth if depth in VALID_DEPTHS else "layer3"

    # --- Grad-CAM for the "to" head ---
    model.zero_grad()
    logits_to[0, to_idx].backward(retain_graph=True)

    activations_to = model.get_gradcam_activations(layer)
    gradients_to = model.get_gradcam_gradients(layer)
    weights_to = gradients_to.mean(dim=(2, 3), keepdim=True)
    cam_to = F.relu((weights_to * activations_to).sum(dim=1, keepdim=True))

    # --- Grad-CAM for the "from" head ---
    model.zero_grad()
    logits_from[0, from_idx].backward()

    activations_from = model.get_gradcam_activations(layer)
    gradients_from = model.get_gradcam_gradients(layer)
    weights_from = gradients_from.mean(dim=(2, 3), keepdim=True)
    cam_from = F.relu((weights_from * activations_from).sum(dim=1, keepdim=True))

    # --- Merge both heatmaps ---
    cam = (cam_from + cam_to) / 2.0  # (1, 1, 8, 8)

    # --- Contrast thresholding: suppress background noise ---
    positive_mask = cam > 0
    if positive_mask.any():
        mean_val = cam[positive_mask].mean()
        cam = cam * (cam >= mean_val).float()  # zero out below-mean squares
        cam = cam ** 2                          # square to make clusters pop

    # --- Diffuse the heatmap ---
    eps = 1e-8
    max_val = cam.max().item()

    if max_val <= eps or not math.isfinite(max_val):
        # Fallback: highlight only the from/to squares
        # Convert python-chess indices (a1=0) to display indices (a8=0)
        fallback = [MIN_HEAT_FLOOR] * 64
        fallback[_sq_to_display(from_idx)] = 1.0
        fallback[_sq_to_display(to_idx)] = 1.0
        return {"weights": fallback, "top_moves": top_moves}

    # Temperature scaling — flatten the distribution before normalizing
    cam = cam / (TEMPERATURE + eps)

    # Gaussian blur — bleed heat into neighboring squares
    if GAUSSIAN_SIGMA > 0:
        k = 3  # 3×3 kernel covers immediate neighbors
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * GAUSSIAN_SIGMA**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)
        cam = F.conv2d(cam, kernel, padding=k // 2)

    # Normalize to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + eps)

    # Convert to numpy for sqrt + floor
    # Flip vertically: tensor row 0 = rank 1 (a1), but frontend expects row 0 = rank 8 (a8)
    heatmap = np.flipud(cam.squeeze().detach().numpy())

    # Square root — boost faint activations so they show as a visible glow
    heatmap = np.sqrt(heatmap)

    # Min-heat floor — every square gets at least a faint baseline
    heatmap = np.clip(heatmap * (1.0 - MIN_HEAT_FLOOR) + MIN_HEAT_FLOOR, 0.0, 1.0)

    return {"weights": [float(v) for v in heatmap.flatten()], "top_moves": top_moves}
