# AutoTender-VN — Dedicated Embedding Model Serving Container
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Cài git (cần để clone deepx-embed) và curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Dùng uv để cài đặt siêu nhanh và chống lỗi timeout mạng/hash mismatch
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.3.1+cpu" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.30.0" \
    "pydantic>=2.7.0" \
    "sentence-transformers>=3.0.0" \
    "einops>=0.8.0" \
    "git+https://github.com/dx-tech-ai/deepx-embed.git"

COPY server.py .

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

