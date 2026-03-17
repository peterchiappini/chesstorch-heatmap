FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only (much smaller than full torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py gradcam.py model.py board_encoder.py ./

# Copy model weights and metadata
COPY weights/ weights/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
