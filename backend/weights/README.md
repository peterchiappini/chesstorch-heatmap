# Weights Directory

Trained model weights (`.pth` files) are excluded from git due to size (~12MB each).

## What's in git
- `*.json` — Training metadata (epochs, accuracy, loss, positions count)

## Regenerate weights
After downloading the data, train each model:
```bash
cd backend
python3 train.py carlsen --epochs 10
python3 train.py tal --epochs 15
# etc.
```

Or train all available datasets:
```bash
for key in $(ls data/*.pgn | xargs -I{} basename {} .pgn); do
  python3 train.py $key --epochs 10
done
```
