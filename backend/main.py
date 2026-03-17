import json
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gradcam import generate_heatmap
from model import ChessCNN, get_device

app = FastAPI()

# Allow configurable CORS origins for production
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ──
# Scan available weights at startup (just metadata), load models on first request.
device = get_device()
models: dict[str, ChessCNN] = {}
model_meta: dict[str, dict] = {}

weights_dir = Path(__file__).parent / "weights"
data_dir = Path(__file__).parent / "data"

available: list[str] = []

if weights_dir.is_dir():
    for pth_file in weights_dir.glob("*.pth"):
        name = pth_file.stem
        available.append(name)

        # Load metadata only (not the model itself)
        meta = {}
        data_meta_path = data_dir / f"{name}.json"
        if data_meta_path.exists():
            with open(data_meta_path) as f:
                meta.update(json.load(f))

        train_meta_path = pth_file.with_suffix(".json")
        if train_meta_path.exists():
            with open(train_meta_path) as f:
                meta.update(json.load(f))

        model_meta[name] = meta

available.sort()
print(f"Available personalities: {available} (lazy loading)")


def _get_model(name: str) -> ChessCNN:
    """Load a model on first request, then cache it."""
    if name in models:
        return models[name]

    pth_path = weights_dir / f"{name}.pth"
    if not pth_path.exists():
        raise HTTPException(status_code=404, detail=f"No weights for '{name}'")

    print(f"Loading model: {name} ...")
    m = ChessCNN()
    m.load_state_dict(torch.load(pth_path, map_location=device))
    m.eval()
    m.install_gradcam_hooks()
    models[name] = m
    print(f"  {name} loaded")
    return m


class ChessRequest(BaseModel):
    fen: str
    personality: str = "carlsen"
    depth: str = "layer3"


@app.get("/api/personalities")
async def get_personalities():
    result = []
    for name in available:
        meta = model_meta.get(name, {})
        player = meta.get("player", name.capitalize())
        tc = meta.get("time_control", "")
        label = f"{player} ({tc})" if tc else player
        result.append({
            "key": name,
            "label": label,
            "player": player,
            "time_control": tc,
            "source": meta.get("source", ""),
            "games": meta.get("games", 0),
            "epochs": meta.get("epochs", 0),
            "accuracy": meta.get("final_accuracy", 0),
            "loss": meta.get("final_loss", 0),
            "positions": meta.get("positions", 0),
            "description": meta.get("description", ""),
        })
    return {"personalities": result}


@app.post("/api/heatmap")
async def get_heatmap(request: ChessRequest):
    if request.personality not in available:
        raise HTTPException(status_code=404, detail=f"Unknown personality '{request.personality}'. Available: {available}")
    model = _get_model(request.personality)
    try:
        result = generate_heatmap(model, request.fen, depth=request.depth)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
