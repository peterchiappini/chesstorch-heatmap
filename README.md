# Chess AI Heatmap

An interactive web app that visualizes the "attention" of a chess AI using heatmaps overlaid on a chessboard. Train CNN models on Grandmaster games, tactical puzzles, or any PGN file — then watch the neural network change its mind as you move pieces.

**Live demo:** https://chesstorch-heatmap.web.app

---

## How it works

Each personality is a small CNN trained on a specific player's games or a dataset (puzzles, master games, etc.). When you make a move, the board is encoded as a 17×8×8 tensor and passed through the network. [Grad-CAM](https://arxiv.org/abs/1610.02391) traces the decision backwards to highlight which squares influenced the predicted move. A Stockfish 18 engine (WebAssembly) runs in parallel for the eval bar and best-move arrow.

---

## Local setup

### Prerequisites

- Node.js 18+
- Python 3.10+

### 1. Frontend

```bash
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and by default points to `http://localhost:8000` for the backend API. To point at a different backend set `VITE_API_URL` in a `.env.local` file:

```
VITE_API_URL=http://localhost:8000
```

### 2. Stockfish (required for eval bar)

The Stockfish WebAssembly files are not in the repo due to size (~108MB). Copy them into `public/`:

```bash
# Option A — copy from the npm package
npm install stockfish
cp node_modules/stockfish/src/stockfish.js public/
cp node_modules/stockfish/src/stockfish.wasm public/

# Option B — download directly
curl -L https://github.com/lichess-org/stockfish.wasm/releases/latest/download/stockfish.js -o public/stockfish.js
curl -L https://github.com/lichess-org/stockfish.wasm/releases/latest/download/stockfish.wasm -o public/stockfish.wasm
```

Without these files the eval bar and Stockfish personality won't work, but the CNN heatmap personalities will.

### 3. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Download training data & train a model

```bash
cd backend
source venv/bin/activate

# Download a player's games (e.g. Carlsen from pgnmentor.com)
python download_data.py player carlsen

# Train
python train.py carlsen --epochs 10
```

The trained weights are saved to `backend/weights/carlsen.pth`. Start the API server and the personality will appear automatically:

```bash
uvicorn main:app --reload
```

Other data sources:

```bash
# Combined GM dataset (10k games from 20+ grandmasters)
python download_data.py master

# Lichess tactical puzzles (requires the puzzle CSV — see below)
python download_data.py puzzles --csv /path/to/lichess_db_puzzle.csv

# Mix sources (e.g. 10k puzzles + 4k master games)
python download_data.py mix puzzles:10000 master:4000 --output puzzles_master

# See all available player personalities
python download_data.py list
```

#### Lichess puzzle database

Download from https://database.lichess.org/#puzzles (`lichess_db_puzzle.csv.zst`), then decompress:

```bash
zstd -d lichess_db_puzzle.csv.zst
python download_data.py puzzles --csv lichess_db_puzzle.csv --min-rating 1200 --max-rating 2000
python train.py puzzles --epochs 10
```

---

## Project structure

```
├── src/                  # React + TypeScript frontend
│   ├── App.tsx           # Main app component
│   └── useStockfish.ts   # Stockfish WebAssembly hook
├── public/               # Static assets (stockfish.js / .wasm go here — not in git)
├── backend/
│   ├── main.py           # FastAPI server
│   ├── model.py          # CNN architecture (ChessCNN)
│   ├── board_encoder.py  # FEN → 17×8×8 tensor encoding
│   ├── gradcam.py        # Grad-CAM heatmap generation
│   ├── train.py          # Training loop
│   ├── download_data.py  # Data pipeline (pgnmentor, chess.com, Lichess, puzzles)
│   ├── data/             # PGN files and dataset metadata (PGNs gitignored)
│   └── weights/          # Trained weights — .pth files gitignored, .json metadata tracked
```

---

## Adding a new personality

1. Get a PGN file (any source)
2. `python train.py --pgn your_file.pgn --weights weights/your_name.pth`
3. Restart the backend — it auto-discovers any `.pth` in `weights/`

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React, TypeScript, Vite, `react-chessboard`, `chess.js` |
| Backend | Python, FastAPI, PyTorch |
| Engine | Stockfish 18 (WebAssembly, runs in-browser) |
| Explainability | Grad-CAM |
| Hosting | Firebase Hosting (frontend), Google Cloud Run (backend) |
