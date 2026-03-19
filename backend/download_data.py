"""
Unified data download pipeline for chess training datasets.

Commands:
    python download_data.py master                        # GM classical from pgnmentor
    python download_data.py beginner                      # sub-1000 from Lichess dump
    python download_data.py player carlsen_classical      # single personality
    python download_data.py player carlsen_blitz
    python download_data.py player all                    # all personalities
    python download_data.py list                          # show available personalities
    python download_data.py all                           # master + beginner
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import time
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GM_DIR = os.path.join(DATA_DIR, "gm")

# ── pgnmentor.com GMs for master dataset ──
GM_PLAYERS = [
    "Carlsen", "Nakamura", "Aronian", "Caruana", "Firouzja",
    "Nepomniachtchi", "Anand", "Kramnik", "Kasparov", "Tal",
]
_PGN_MENTOR_URL = "https://www.pgnmentor.com/players/{name}.zip"

# ── Personality definitions: {key: {...}} ──
# All from pgnmentor.com (classical OTB tournament games)
PERSONALITIES: dict[str, dict] = {
    # Modern elite
    "carlsen":    {"source": "pgnmentor", "pgnmentor": "Carlsen",         "player": "Carlsen",         "time_control": "All"},
    "hikaru":     {"source": "pgnmentor", "pgnmentor": "Nakamura",        "player": "Nakamura",        "time_control": "All"},
    "firouzja":   {"source": "pgnmentor", "pgnmentor": "Firouzja",        "player": "Firouzja",        "time_control": "All"},
    "nepo":       {"source": "pgnmentor", "pgnmentor": "Nepomniachtchi",  "player": "Nepomniachtchi",  "time_control": "All"},
    "caruana":    {"source": "pgnmentor", "pgnmentor": "Caruana",          "player": "Caruana",         "time_control": "All"},
    "aronian":    {"source": "pgnmentor", "pgnmentor": "Aronian",          "player": "Aronian",         "time_control": "All"},
    "ding":       {"source": "pgnmentor", "pgnmentor": "Ding",             "player": "Ding Liren",      "time_control": "All"},
    "giri":       {"source": "pgnmentor", "pgnmentor": "Giri",             "player": "Giri",            "time_control": "All"},
    "topalov":    {"source": "pgnmentor", "pgnmentor": "Topalov",          "player": "Topalov",         "time_control": "All"},
    "ivanchuk":   {"source": "pgnmentor", "pgnmentor": "Ivanchuk",         "player": "Ivanchuk",        "time_control": "All"},

    # World Champions (historical)
    "kasparov":   {"source": "pgnmentor", "pgnmentor": "Kasparov",    "player": "Kasparov",    "time_control": "All"},
    "karpov":     {"source": "pgnmentor", "pgnmentor": "Karpov",      "player": "Karpov",      "time_control": "All"},
    "fischer":    {"source": "pgnmentor", "pgnmentor": "Fischer",     "player": "Fischer",     "time_control": "All"},
    "tal":        {"source": "pgnmentor", "pgnmentor": "Tal",         "player": "Tal",         "time_control": "All"},
    "spassky":    {"source": "pgnmentor", "pgnmentor": "Spassky",     "player": "Spassky",     "time_control": "All"},
    "petrosian":  {"source": "pgnmentor", "pgnmentor": "Petrosian",   "player": "Petrosian",   "time_control": "All"},
    "botvinnik":  {"source": "pgnmentor", "pgnmentor": "Botvinnik",   "player": "Botvinnik",   "time_control": "All"},
    "smyslov":    {"source": "pgnmentor", "pgnmentor": "Smyslov",     "player": "Smyslov",     "time_control": "All"},
    "kramnik":    {"source": "pgnmentor", "pgnmentor": "Kramnik",     "player": "Kramnik",     "time_control": "All"},
    "anand":      {"source": "pgnmentor", "pgnmentor": "Anand",       "player": "Anand",       "time_control": "All"},

    # Legends
    "capablanca": {"source": "pgnmentor", "pgnmentor": "Capablanca",  "player": "Capablanca",  "time_control": "All"},
    "alekhine":   {"source": "pgnmentor", "pgnmentor": "Alekhine",    "player": "Alekhine",    "time_control": "All"},
    "morphy":     {"source": "pgnmentor", "pgnmentor": "Morphy",      "player": "Morphy",      "time_control": "All"},
    "euwe":       {"source": "pgnmentor", "pgnmentor": "Euwe",        "player": "Euwe",        "time_control": "All"},

    # Chess.com blitz/bullet (modern GMs with huge game counts)
    "carlsen_blitz":   {"source": "chesscom", "username": "MagnusCarlsen",   "player": "Carlsen",         "time_control": "Blitz",  "tc_filter": "blitz"},
    "carlsen_bullet":  {"source": "chesscom", "username": "MagnusCarlsen",   "player": "Carlsen",         "time_control": "Bullet", "tc_filter": "bullet"},
    "hikaru_blitz":    {"source": "chesscom", "username": "Hikaru",          "player": "Nakamura",        "time_control": "Blitz",  "tc_filter": "blitz"},
    "hikaru_bullet":   {"source": "chesscom", "username": "Hikaru",          "player": "Nakamura",        "time_control": "Bullet", "tc_filter": "bullet"},
    "danya_blitz":     {"source": "chesscom", "username": "DanielNaroditsky","player": "Naroditsky",      "time_control": "Blitz",  "tc_filter": "blitz"},
    "danya_bullet":    {"source": "chesscom", "username": "DanielNaroditsky","player": "Naroditsky",      "time_control": "Bullet", "tc_filter": "bullet"},
    "caruana_blitz":   {"source": "chesscom", "username": "FabianoCaruana",  "player": "Caruana",         "time_control": "Blitz",  "tc_filter": "blitz"},
    "nepo_blitz":      {"source": "chesscom", "username": "lachesisQ",       "player": "Nepomniachtchi",  "time_control": "Blitz",  "tc_filter": "blitz"},
}

_LICHESS_API_URL = "https://lichess.org/api/games/user/{username}?max={max_games}&perfType={perfType}"
_CHESSCOM_ARCHIVES_URL = "https://api.chess.com/pub/player/{username}/games/archives"

# ── Fast PGN header scanning ──
_RE_WHITE_ELO = re.compile(r'\[WhiteElo "(\d+)"\]')
_RE_BLACK_ELO = re.compile(r'\[BlackElo "(\d+)"\]')
_RE_EVENT = re.compile(r'\[Event "([^"]+)"\]')


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

def download_pgnmentor(name: str, dest_dir: str = GM_DIR, zip_name: str | None = None) -> str | None:
    """Download a pgnmentor zip, extract .pgn files, return local path.

    Args:
        name: Local filename stem (e.g. "carlsen" → "carlsen.pgn")
        dest_dir: Directory to save into
        zip_name: pgnmentor zip name if different from name (e.g. "Carlsen")
    """
    os.makedirs(dest_dir, exist_ok=True)
    pgn_path = os.path.join(dest_dir, f"{name}.pgn")

    if os.path.exists(pgn_path):
        print(f"    {name}: already exists, skipping")
        return pgn_path

    url = _PGN_MENTOR_URL.format(name=zip_name or name)
    print(f"    {name}: downloading ...")

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (chesstorch-heatmap)",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"    {name}: FAILED ({e})")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            pgn_files = [f for f in zf.namelist() if f.lower().endswith(".pgn")]
            if not pgn_files:
                print(f"    {name}: zip has no .pgn files")
                return None
            with open(pgn_path, "w") as out:
                for pf in pgn_files:
                    out.write(zf.read(pf).decode("utf-8", errors="replace"))
                    out.write("\n")
    except zipfile.BadZipFile:
        print(f"    {name}: bad zip")
        return None

    with open(pgn_path) as f:
        count = sum(1 for line in f if line.startswith("[Event "))
    print(f"    {name}: {count:,} games")
    return pgn_path


def _get_lichess_token() -> str | None:
    """Read Lichess API token from LICHESS_TOKEN env var or backend/.lichess-token file."""
    token = os.environ.get("LICHESS_TOKEN")
    if token:
        return token
    token_path = os.path.join(os.path.dirname(__file__), ".lichess-token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return None


def download_lichess(username: str, output_path: str, perf_type: str, max_games: int = 5000) -> int:
    """Download games from Lichess API with a time control filter, return game count.

    Handles 429 rate limits with exponential backoff (up to 3 retries).
    """
    url = _LICHESS_API_URL.format(username=username, max_games=max_games, perfType=perf_type)
    headers = {"Accept": "application/x-chess-pgn"}
    token = _get_lichess_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print(f"    (using Lichess API token)")

    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            games = 0
            with urllib.request.urlopen(req, timeout=600) as response, open(output_path, "w") as out:
                for raw_line in response:
                    line = raw_line.decode("utf-8")
                    out.write(line)
                    if line.startswith("[Event "):
                        games += 1
                        if games % 500 == 0:
                            print(f"    {games} games ...")
            return games
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited (429). Waiting {wait}s before retry ...")
                time.sleep(wait)
            else:
                raise
    return 0


def _classify_tc(tc_header: str) -> str:
    """Classify a chess.com TimeControl header into bullet/blitz/rapid/classical."""
    try:
        # Format is "base" or "base+increment"
        base = int(tc_header.split("+")[0])
    except (ValueError, IndexError):
        return "unknown"
    if base < 180:
        return "bullet"
    if base < 600:
        return "blitz"
    if base < 1800:
        return "rapid"
    return "classical"


_RE_TIMECONTROL = re.compile(r'\[TimeControl "([^"]+)"\]')


def download_chesscom(username: str, output_path: str, tc_filter: str, max_games: int = 5000) -> int:
    """Download games from chess.com, filtered by time control category.

    Args:
        username: chess.com username
        output_path: where to save the PGN
        tc_filter: "bullet", "blitz", "rapid", or "classical"
        max_games: max games to collect
    """
    # Get list of monthly archives
    archives_url = _CHESSCOM_ARCHIVES_URL.format(username=username)
    req = urllib.request.Request(archives_url, headers={
        "User-Agent": "Mozilla/5.0 (chesstorch-heatmap)",
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    Failed to fetch archives: {e}")
        return 0

    archive_urls = data.get("archives", [])
    # Process newest months first for most recent games
    archive_urls.reverse()
    print(f"    {len(archive_urls)} monthly archives available")

    games = 0
    with open(output_path, "w") as out:
        for archive_url in archive_urls:
            if games >= max_games:
                break

            pgn_url = archive_url + "/pgn"
            try:
                req = urllib.request.Request(pgn_url, headers={
                    "User-Agent": "Mozilla/5.0 (chesstorch-heatmap)",
                })
                with urllib.request.urlopen(req, timeout=60) as resp:
                    month_pgn = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                print(f"    Skipping {archive_url}: {e}")
                continue

            # Split into game blocks and filter by time control
            current: list[str] = []
            for line in month_pgn.splitlines(keepends=True):
                if line.startswith("[Event ") and current:
                    block = "".join(current)
                    tc_match = _RE_TIMECONTROL.search(block)
                    if tc_match and _classify_tc(tc_match.group(1)) == tc_filter:
                        out.write(block)
                        if not block.endswith("\n"):
                            out.write("\n")
                        games += 1
                        if games >= max_games:
                            break
                    current = []
                current.append(line)

            # Last block in month
            if current and games < max_games:
                block = "".join(current)
                tc_match = _RE_TIMECONTROL.search(block)
                if tc_match and _classify_tc(tc_match.group(1)) == tc_filter:
                    out.write(block)
                    games += 1

            if games % 500 < 50:
                print(f"    {games:,} {tc_filter} games so far ...")

    print(f"    {games:,} {tc_filter} games collected")
    return games


def _collect_blocks(path: str) -> list[str]:
    """Read a PGN file into a list of raw game text blocks."""
    blocks: list[str] = []
    current: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith("[Event ") and current:
                blocks.append("".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append("".join(current))
    return blocks


# ═══════════════════════════════════════════════════════════════════
#  MASTER: combined GM classical dataset
# ═══════════════════════════════════════════════════════════════════

def download_master(target: int = 10_000, skip_download: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    master_path = os.path.join(DATA_DIR, "master.pgn")

    # Collect all pgnmentor personality PGN files
    gm_files: list[str] = []
    pgnmentor_keys = [k for k, v in PERSONALITIES.items() if v["source"] == "pgnmentor"]

    if not skip_download:
        print("Downloading GM games from pgnmentor.com ...")
        for key in pgnmentor_keys:
            download_player(key)

    for key in pgnmentor_keys:
        path = os.path.join(DATA_DIR, f"{key}.pgn")
        if os.path.exists(path):
            gm_files.append(path)

    print(f"Found {len(gm_files)} GM files")

    if not gm_files:
        print("No GM files available.")
        return

    print("Collecting game blocks ...")
    blocks: list[str] = []
    for path in gm_files:
        blocks.extend(_collect_blocks(path))

    print(f"  Total: {len(blocks):,} games across {len(gm_files)} GMs")

    if len(blocks) > target:
        blocks = random.sample(blocks, target)

    with open(master_path, "w") as out:
        for block in blocks:
            out.write(block)
            if not block.endswith("\n"):
                out.write("\n")

    # Save metadata
    gm_names = [PERSONALITIES[k]["player"] for k in pgnmentor_keys if os.path.exists(os.path.join(DATA_DIR, f"{k}.pgn"))]
    meta = {
        "dataset": "master",
        "games": len(blocks),
        "sampled_from": len(blocks) if len(blocks) <= target else f"{len(blocks):,} total across {len(gm_files)} GMs",
        "source": "pgnmentor.com",
        "players": gm_names,
        "description": (
            f"Combined GM dataset: {len(blocks):,} games randomly sampled from "
            f"{len(gm_files)} Grandmasters ({', '.join(gm_names[:5])}, and {len(gm_names)-5} more). "
            f"Sourced from pgnmentor.com. Includes OTB tournament games across all time controls."
        ),
    }
    with open(os.path.join(DATA_DIR, "master.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Master dataset: {len(blocks):,} games → {master_path}")


# ═══════════════════════════════════════════════════════════════════
#  BEGINNER: reservoir-sampled from Lichess DB dump
# ═══════════════════════════════════════════════════════════════════

def download_beginner(target: int = 10_000, db_path: str | None = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    beginner_path = os.path.join(DATA_DIR, "beginner.pgn")

    if os.path.exists(beginner_path):
        with open(beginner_path) as f:
            existing = sum(1 for line in f if line.startswith("[Event "))
        if existing >= target:
            print(f"beginner.pgn already has {existing:,} games (target: {target:,}), skipping.")
            return

    if db_path is None:
        for f in os.listdir(DATA_DIR):
            full = os.path.join(DATA_DIR, f)
            if f.endswith(".pgn") and f not in ("master.pgn", "beginner.pgn"):
                if os.path.getsize(full) > 100_000_000:
                    db_path = full
                    break

    if db_path is None or not os.path.exists(db_path):
        print("No Lichess DB dump found.")
        print("  Download one from https://database.lichess.org/")
        print("  Place it in backend/data/ and rerun.")
        return

    print(f"Reservoir-sampling beginner games from {os.path.basename(db_path)} ...")

    reservoir: list[str] = []
    eligible = 0
    scanned = 0
    t0 = time.time()

    with open(db_path) as f:
        current: list[str] = []
        for line in f:
            if line.startswith("[Event ") and current:
                scanned += 1
                block_text = "".join(current)
                current = []

                event_match = _RE_EVENT.search(block_text)
                if event_match:
                    event = event_match.group(1)
                    if "Bullet" in event or "UltraBullet" in event:
                        current.append(line)
                        continue

                wm = _RE_WHITE_ELO.search(block_text)
                bm = _RE_BLACK_ELO.search(block_text)
                if wm and bm and int(wm.group(1)) < 1000 and int(bm.group(1)) < 1000:
                    eligible += 1
                    if len(reservoir) < target:
                        reservoir.append(block_text)
                    else:
                        j = random.randint(0, eligible - 1)
                        if j < target:
                            reservoir[j] = block_text

                if scanned % 100_000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {scanned:>10,} scanned ({elapsed:.0f}s) — {eligible:,} eligible, {len(reservoir):,} in reservoir")

            current.append(line)

    elapsed = time.time() - t0
    random.shuffle(reservoir)
    print(f"  Done: {scanned:,} scanned, {eligible:,} eligible, {len(reservoir):,} sampled ({elapsed:.1f}s)")

    with open(beginner_path, "w") as out:
        for block in reservoir:
            out.write(block)
            if not block.endswith("\n"):
                out.write("\n")

    # Save metadata
    db_name = os.path.basename(db_path)
    meta = {
        "dataset": "beginner",
        "games": len(reservoir),
        "eligible_in_source": eligible,
        "total_scanned": scanned,
        "source": f"lichess.org database dump ({db_name})",
        "elo_filter": "Both players rated < 1000",
        "time_control_filter": "Excludes bullet and ultrabullet",
        "sampling_method": "Reservoir sampling (Algorithm R) — uniform random across the entire file",
        "description": (
            f"{len(reservoir):,} games randomly sampled from {eligible:,} eligible games "
            f"({scanned:,} total scanned) in the Lichess database dump '{db_name}'. "
            f"Both players rated under 1000. Bullet and ultrabullet excluded. "
            f"Reservoir-sampled for uniform distribution across the file."
        ),
    }
    with open(os.path.join(DATA_DIR, "beginner.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Beginner dataset: {len(reservoir):,} games → {beginner_path}")
    dump_size = os.path.getsize(db_path) / (1024**3)
    print(f"\n  Lichess dump ({dump_size:.1f}GB) can now be deleted:")
    print(f"    rm {db_path}")


# ═══════════════════════════════════════════════════════════════════
#  PLAYER: single personality (by key)
# ═══════════════════════════════════════════════════════════════════

def download_player(key: str, max_games: int = 1000) -> None:
    if key == "all":
        last_source = None
        for k in PERSONALITIES:
            src = PERSONALITIES[k]["source"]
            # Pause between API requests to avoid rate limits
            if last_source in ("lichess", "chesscom") and src in ("lichess", "chesscom"):
                print("    (waiting 5s between API requests ...)")
                time.sleep(5)
            download_player(k, max_games)
            last_source = src
        return

    if key not in PERSONALITIES:
        print(f"Unknown personality '{key}'. Run `python download_data.py list` to see options.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    info = PERSONALITIES[key]
    output_path = os.path.join(DATA_DIR, f"{key}.pgn")

    if info["source"] == "pgnmentor":
        pgnmentor_name = info["pgnmentor"]
        print(f"Downloading {info['player']} {info['time_control']} from pgnmentor.com ...")
        # Download directly using the personality key as the filename
        result = download_pgnmentor(key, dest_dir=DATA_DIR, zip_name=pgnmentor_name)
        if not result:
            print(f"  Failed for {key}")
            return
        with open(output_path) as f:
            count = sum(1 for line in f if line.startswith("[Event "))

    elif info["source"] == "lichess":
        username = info["username"]
        perf = info["perfType"]
        print(f"Downloading {info['player']} {info['time_control']} from Lichess ({username}, {perf}) ...")
        count = download_lichess(username, output_path, perf, max_games)

    elif info["source"] == "chesscom":
        username = info["username"]
        tc = info["tc_filter"]
        print(f"Downloading {info['player']} {info['time_control']} from chess.com ({username}, {tc}) ...")
        count = download_chesscom(username, output_path, tc, max_games)

    else:
        print(f"  Unknown source for {key}")
        return

    # Build description based on source
    if info["source"] == "pgnmentor":
        desc = (
            f"OTB tournament games by {info['player']}, sourced from pgnmentor.com. "
            f"Includes all recorded tournament games (classical, rapid, and blitz events). "
            f"{count:,} games total."
        )
    elif info["source"] == "lichess":
        desc = (
            f"Online {info['time_control'].lower()} games by {info['player']} "
            f"(Lichess account: {info['username']}). "
            f"Filtered to {info['perfType']} time control. {count:,} games."
        )
    elif info["source"] == "chesscom":
        desc = (
            f"Online {info['time_control'].lower()} games by {info['player']} "
            f"(chess.com account: {info['username']}). "
            f"Filtered by time control ({info['tc_filter']}). "
            f"Most recent games first. {count:,} games."
        )
    else:
        desc = f"{count:,} games."

    meta = {
        "player": info["player"],
        "time_control": info["time_control"],
        "source": info["source"],
        "games": count,
        "description": desc,
    }
    meta_path = os.path.join(DATA_DIR, f"{key}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done — {count:,} games → {output_path}")


def list_personalities() -> None:
    print(f"\n{'Key':<25} {'Player':<18} {'Time Control':<12} {'Source'}")
    print("─" * 70)
    for key, info in PERSONALITIES.items():
        print(f"{key:<25} {info['player']:<18} {info['time_control']:<12} {info['source']}")
    print()


# ═══════════════════════════════════════════════════════════════════
#  PUZZLES: Lichess tactical puzzle database
# ═══════════════════════════════════════════════════════════════════

def download_puzzles(
    output_dir: str = DATA_DIR,
    csv_path: str | None = None,
    max_puzzles: int = 10_000,
    min_rating: int = 1000,
    max_rating: int = 2500,
    themes: list[str] | None = None,
) -> None:
    """Convert the Lichess puzzle database CSV to a PGN file for training.

    The Lichess puzzle CSV can be downloaded from:
        https://database.lichess.org/#puzzles  (lichess_db_puzzle.csv.zst)

    Decompress it first with:
        zstd -d lichess_db_puzzle.csv.zst

    CSV columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity,
                 NbPlays, Themes, GameUrl, OpeningTags

    The FEN is the board position BEFORE the opponent's forcing move.
    Moves[0] = opponent's move (apply to reach puzzle position).
    Moves[1:] = solution moves (what the model should learn to play).
    """
    import chess
    import chess.pgn
    import csv as csv_mod

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "puzzles.pgn")

    if csv_path is None:
        # Try to auto-detect a puzzle CSV in the data dir
        for fname in os.listdir(output_dir):
            if "puzzle" in fname.lower() and fname.endswith(".csv"):
                csv_path = os.path.join(output_dir, fname)
                print(f"Found puzzle CSV: {csv_path}")
                break

    if csv_path is None or not os.path.exists(csv_path):
        print("No puzzle CSV found.")
        print()
        print("Download the Lichess puzzle database from:")
        print("  https://database.lichess.org/#puzzles")
        print()
        print("Decompress it:")
        print("  zstd -d lichess_db_puzzle.csv.zst")
        print()
        print("Then run:")
        print(f"  python download_data.py puzzles --csv /path/to/lichess_db_puzzle.csv")
        return

    print(f"Reading puzzles from: {csv_path}")
    if min_rating or max_rating:
        print(f"  Rating filter: {min_rating}–{max_rating}")
    if themes:
        print(f"  Theme filter: {', '.join(themes)}")

    written = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f_in, \
         open(output_path, "w") as f_out:

        reader = csv_mod.DictReader(f_in)
        for row in reader:
            if written >= max_puzzles:
                break

            try:
                rating = int(row["Rating"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if rating < min_rating or rating > max_rating:
                skipped += 1
                continue

            if themes:
                row_themes = set(row.get("Themes", "").split())
                if not row_themes.intersection(themes):
                    skipped += 1
                    continue

            puzzle_id = row.get("PuzzleId", str(written))
            fen = row.get("FEN", "")
            moves_uci = row.get("Moves", "").split()

            # Need at least the opponent's setup move + one solution move
            if len(moves_uci) < 2:
                skipped += 1
                continue

            try:
                # Apply the opponent's move to reach the actual puzzle position
                setup_board = chess.Board(fen)
                setup_board.push_uci(moves_uci[0])
                puzzle_fen = setup_board.fen()

                game = chess.pgn.Game()
                game.headers["Event"] = f"Lichess Puzzle {puzzle_id}"
                game.headers["Site"] = f"https://lichess.org/training/{puzzle_id}"
                game.headers["SetUp"] = "1"
                game.headers["FEN"] = puzzle_fen
                game.headers["WhiteElo"] = str(rating)

                node = game
                board = chess.Board(puzzle_fen)
                for uci in moves_uci[1:]:
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    node = node.add_variation(move)
                    board.push(move)

                if not game.next():
                    skipped += 1
                    continue

                print(game, file=f_out, end="\n\n")
                written += 1

            except Exception:
                skipped += 1
                continue

            if written % 1000 == 0:
                print(f"  {written:,} puzzles written ...")

    print(f"Done — {written:,} puzzles → {output_path} ({skipped:,} skipped)")

    # Save metadata
    meta = {
        "player": "Tactical Puzzles",
        "time_control": "Puzzle",
        "source": "lichess_puzzles",
        "games": written,
        "rating_range": f"{min_rating}–{max_rating}",
        "themes": themes or [],
        "description": (
            f"{written:,} Lichess tactical puzzles (rating {min_rating}–{max_rating}). "
            f"Each position has a known best move sequence. "
            f"Trains the model to find forcing tactical moves."
        ),
    }
    meta_path = os.path.join(output_dir, "puzzles.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download chess training data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_master = sub.add_parser("master", help="GM classical games from pgnmentor.com")
    p_master.add_argument("--target", type=int, default=10_000)
    p_master.add_argument("--skip-download", action="store_true")

    p_beginner = sub.add_parser("beginner", help="Sub-1000 games from Lichess DB dump")
    p_beginner.add_argument("--target", type=int, default=10_000)
    p_beginner.add_argument("--db", default=None)

    p_player = sub.add_parser("player", help="Single personality dataset")
    p_player.add_argument("name", help="Personality key (or 'all')")
    p_player.add_argument("--max-games", type=int, default=1000, help="Max games (Lichess only)")

    p_all = sub.add_parser("all", help="Master + beginner datasets")
    p_all.add_argument("--target", type=int, default=10_000)
    p_all.add_argument("--db", default=None)
    p_all.add_argument("--skip-download", action="store_true")

    p_puzzles = sub.add_parser("puzzles", help="Lichess tactical puzzle database")
    p_puzzles.add_argument("--csv", default=None, help="Path to lichess_db_puzzle.csv (decompressed)")
    p_puzzles.add_argument("--max", type=int, default=10_000, dest="max_puzzles", help="Max puzzles to extract")
    p_puzzles.add_argument("--min-rating", type=int, default=1000)
    p_puzzles.add_argument("--max-rating", type=int, default=2500)
    p_puzzles.add_argument("--themes", nargs="+", default=None, help="Filter by theme(s), e.g. --themes fork pin")

    sub.add_parser("list", help="Show available personalities")

    args = parser.parse_args()

    if args.command == "master":
        download_master(args.target, skip_download=args.skip_download)
    elif args.command == "beginner":
        download_beginner(args.target, db_path=args.db)
    elif args.command == "player":
        download_player(args.name, max_games=args.max_games)
    elif args.command == "all":
        download_master(args.target, skip_download=args.skip_download)
        download_beginner(args.target, db_path=args.db)
    elif args.command == "puzzles":
        download_puzzles(
            csv_path=args.csv,
            max_puzzles=args.max_puzzles,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            themes=args.themes,
        )
    elif args.command == "list":
        list_personalities()
