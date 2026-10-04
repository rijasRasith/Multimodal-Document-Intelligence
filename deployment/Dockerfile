FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libmupdf-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY retrieval/ ./retrieval/
COPY api/ ./api/

ENV MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct" \
    ADAPTER_PATH="" \
    MAX_FILE_SIZE_MB="20" \
    PGVECTOR_DSN="" \
    MLFLOW_TRACKING_URI="http://localhost:5000" \
    API_KEY="change-me-in-production"

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
