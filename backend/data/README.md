# Data Directory

This directory contains training datasets for the ChessTorch Heatmap models. The large files (PGN games, cached tensors) are excluded from git due to size.

## What's in git
- `*.json` — Metadata for each dataset (source, game count, description)

## What's NOT in git (regenerate locally)

### PGN game files (`*.pgn`)
Download with:
```bash
cd backend
python3 download_data.py player all          # GM personalities (~24 players from pgnmentor.com + chess.com)
python3 download_data.py master              # Combined 10k GM dataset
python3 download_data.py beginner            # 10k sub-1000 Elo (requires Lichess DB dump)
```

### Cached tensors (`*.pt`)
Auto-generated on first training run, or pre-cache all at once:
```bash
python3 cache_datasets.py
```

### Lichess database dump
Required only for beginner dataset extraction. Download from https://database.lichess.org/, place in this directory, then run `python3 download_data.py beginner`.

## Sources
- **pgnmentor.com** — OTB tournament games for 24 Grandmasters (Carlsen, Kasparov, Fischer, Tal, etc.)
- **chess.com API** — Online blitz/bullet games for modern GMs
- **lichess.org database** — Beginner games (both players < 1000 Elo)
