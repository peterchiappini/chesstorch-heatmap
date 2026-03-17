"""
Pre-cache all PGN datasets into .pt tensor files in parallel.

Run this before training to convert all PGNs upfront using multiple CPU cores.
Training will then skip parsing entirely and load tensors instantly.

Usage:
    python cache_datasets.py              # cache all .pgn files in data/
    python cache_datasets.py --workers 4  # limit parallelism
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def cache_one(pgn_path: str) -> str:
    """Parse and cache a single PGN file. Returns a status string."""
    from board_encoder import load_or_parse_pgn

    name = os.path.basename(pgn_path).replace(".pgn", "")
    cache_path = pgn_path.replace(".pgn", ".pt")

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(pgn_path):
        return f"  {name}: already cached"

    try:
        t0 = time.time()
        _, _, _, num_games = load_or_parse_pgn(pgn_path)
        elapsed = time.time() - t0
        return f"  {name}: {num_games:,} games cached ({elapsed:.1f}s)"
    except Exception as e:
        return f"  {name}: FAILED ({e})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache PGN datasets into .pt files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    pgn_files = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".pgn") and f not in ("master.pgn", "beginner.pgn")
    )

    # Also include master and beginner if they exist
    for name in ("master.pgn", "beginner.pgn"):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            pgn_files.append(path)

    if not pgn_files:
        print("No PGN files found in data/")
        sys.exit(1)

    uncached = []
    for p in pgn_files:
        cache_path = p.replace(".pgn", ".pt")
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(p):
            continue
        uncached.append(p)

    print(f"Found {len(pgn_files)} PGN files, {len(uncached)} need caching")
    if not uncached:
        print("All datasets already cached!")
        return

    workers = min(args.workers, len(uncached))
    print(f"Caching with {workers} parallel workers ...\n")

    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(workers)
    try:
        results = pool.map(cache_one, uncached)
        for r in results:
            print(r)
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
