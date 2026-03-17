"""
Chess data pipeline: board encoding, PGN parsing, and PyTorch Dataset.

Converts chess positions into 17×8×8 tensors suitable for CNN training
and provides a Dataset class for use with torch DataLoaders.
"""

from __future__ import annotations

import io
import logging
from typing import Iterator

import chess
import chess.pgn

# Suppress chess.pgn warnings about illegal moves in corrupted games
logging.getLogger("chess.pgn").setLevel(logging.ERROR)
import numpy as np
import torch
from torch.utils.data import Dataset

# Piece-type to channel index. chess.PAWN == 1 … chess.KING == 6,
# so subtract 1 to get a 0-based channel offset.
_WHITE_CHANNEL_OFFSET = 0  # channels 0–5
_BLACK_CHANNEL_OFFSET = 6  # channels 6–11

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board state as a (17, 8, 8) float tensor.

    Channels:
        0–5   White pieces  (P, N, B, R, Q, K)
        6–11  Black pieces  (p, n, b, r, q, k)
        12    White kingside  castling right
        13    White queenside castling right
        14    Black kingside  castling right
        15    Black queenside castling right
        16    Side to move    (1 = White, 0 = Black)
    """
    planes = np.zeros((17, 8, 8), dtype=np.float32)

    # --- piece planes (channels 0–11) ---
    for piece_type in _PIECE_TYPES:
        channel = piece_type - 1  # PAWN=1 → channel 0, etc.
        for sq in board.pieces(piece_type, chess.WHITE):
            rank, file = divmod(sq, 8)
            planes[_WHITE_CHANNEL_OFFSET + channel, rank, file] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            rank, file = divmod(sq, 8)
            planes[_BLACK_CHANNEL_OFFSET + channel, rank, file] = 1.0

    # --- castling rights (channels 12–15) ---
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0

    # --- side to move (channel 16) ---
    if board.turn == chess.WHITE:
        planes[16] = 1.0

    return torch.from_numpy(planes)


def parse_pgn(pgn_string_or_file: str | io.TextIOBase) -> Iterator[tuple[torch.Tensor, tuple[int, int]]]:
    """Parse PGN data and yield (board_tensor, (from_square, to_square)) for every move.

    Args:
        pgn_string_or_file: Either a PGN string or an open text file handle.

    Yields:
        (tensor, (from_square, to_square)) where tensor is (17, 8, 8) and
        both square indices are 0–63.
    """
    if isinstance(pgn_string_or_file, str):
        pgn_string_or_file = io.StringIO(pgn_string_or_file)

    while True:
        game = chess.pgn.read_game(pgn_string_or_file)
        if game is None:
            break

        board = game.board()
        for move in game.mainline_moves():
            tensor = board_to_tensor(board)
            yield tensor, (move.from_square, move.to_square)
            board.push(move)


class ChessDataset(Dataset[tuple[torch.Tensor, tuple[int, int]]]):
    """PyTorch Dataset wrapping a list of (board_tensor, (from_sq, to_sq)) pairs."""

    def __init__(self, samples: list[tuple[torch.Tensor, tuple[int, int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor, (from_sq, to_sq) = self.samples[idx]
        return tensor, torch.tensor(from_sq, dtype=torch.long), torch.tensor(to_sq, dtype=torch.long)


def load_or_parse_pgn(pgn_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load cached tensors from a .pt file, or parse the PGN and cache the result.

    Returns:
        (tensors, from_squares, to_squares, num_games)
    """
    import os

    cache_path = pgn_path.replace(".pgn", ".pt")

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(pgn_path):
        print(f"Loading cached tensors: {cache_path}")
        cached = torch.load(cache_path, map_location="cpu")
        tensors = cached["tensors"]
        from_squares = cached["from_squares"]
        to_squares = cached["to_squares"]
        num_games = cached["num_games"]
        print(f"  {num_games:,} games, {len(tensors):,} positions (cached)")
        return tensors, from_squares, to_squares, num_games

    print(f"Parsing PGN: {pgn_path}")
    with open(pgn_path) as f:
        num_games = sum(1 for line in f if line.startswith("[Event "))
    with open(pgn_path) as f:
        samples = list(parse_pgn(f))
    print(f"  {num_games:,} games, {len(samples):,} positions")

    if len(samples) == 0:
        raise ValueError(f"No positions found in {pgn_path}")

    tensors = torch.stack([s[0] for s in samples])
    from_squares = torch.tensor([s[1][0] for s in samples], dtype=torch.long)
    to_squares = torch.tensor([s[1][1] for s in samples], dtype=torch.long)

    print(f"  Caching to {cache_path} ...")
    torch.save({
        "tensors": tensors,
        "from_squares": from_squares,
        "to_squares": to_squares,
        "num_games": num_games,
    }, cache_path)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Cached ({size_mb:.0f}MB)")

    return tensors, from_squares, to_squares, num_games
