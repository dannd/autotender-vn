"""AutoTender-VN Dedicated Embedding Model Serving Microservice.

Chạy trong Docker container độc lập (port 8080).
Hỗ trợ cả:
1. deepx-embedding-v1 (qua thư viện deepx_embed) — 8K context, linear attention.
2. bkai-foundation-models/vietnamese-bi-encoder (qua sentence_transformers) — fallback.

API:
- GET  /health -> {"status": "ok", "model": ...}
- GET  /info   -> Thông tin model, vector size, context length
- POST /embed  -> {"texts": ["câu 1", "câu 2"], "dimension": 1024} -> {"embeddings": [[...], [...]]}
"""

import os
from typing import Any
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AutoTender Embedding Serving Service",
    description="Microservice phục vụ vector hóa văn bản pháp lý (deepx-embedding-v1 / vi_bi_encoder)",
    version="1.0.0",
)

import torch
num_threads = min(os.cpu_count() or 8, 12)
try:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
except Exception:
    pass

MODEL_KEY = os.environ.get("EMBEDDING_MODEL_KEY", "deepx_v1")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "dxtech-asia/deepx-embedding-v1")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "1024"))

_model = None
_model_type = None


def get_model():
    global _model, _model_type
    if _model is not None:
        return _model, _model_type

    print(f"Loading embedding model '{MODEL_NAME}' (key: {MODEL_KEY})...")
    
    # 1. Thử load qua deepx_embed nếu là deepx_v1
    if "deepx" in MODEL_KEY.lower() or "deepx" in MODEL_NAME.lower():
        try:
            import torch
            from deepx_embed import DeepXEmbed
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = DeepXEmbed.from_pretrained(MODEL_NAME, device=device)
            _model_type = "deepx_embed"
            print(f"Loaded {MODEL_NAME} successfully via deepx_embed (device={device}).")
            return _model, _model_type
        except Exception as e:
            print(f"Warning: Cannot load via deepx_embed: {e}. Falling back to SentenceTransformer...")

    # 2. Fallback sang sentence-transformers
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME)
    _model_type = "sentence_transformers"
    print(f"Loaded {MODEL_NAME} successfully via SentenceTransformer.")
    return _model, _model_type


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="Danh sách các đoạn văn bản cần vector hóa")
    dimension: int | None = Field(None, description="Chiều vector mong muốn (Matryoshka)")


class EmbedResponse(BaseModel):
    model: str
    model_type: str
    dimension: int
    count: int
    embeddings: list[list[float]]


@app.get("/")
def root():
    return {
        "service": "AutoTender Embedding Model Server",
        "model": MODEL_NAME,
        "endpoints": ["/health", "/info", "/embed"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "model_key": MODEL_KEY}


@app.get("/info")
def info():
    return {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "default_vector_size": VECTOR_SIZE,
        "context_length": 8192 if "deepx" in MODEL_KEY else 256,
        "architecture": "Linear Attention (Gated DeltaNet-2)" if "deepx" in MODEL_KEY else "Transformer Bi-Encoder",
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Danh sách texts không được rỗng.")

    model, model_type = get_model()
    try:
        dim_target = req.dimension or VECTOR_SIZE
        if model_type == "deepx_embed":
            # deepx_embed encode with Matryoshka dimension (e.g. 1024)
            vecs = model.encode(req.texts, truncate_dim=dim_target)
            if isinstance(vecs, np.ndarray):
                embeddings = vecs.tolist()
            else:
                embeddings = [v.tolist() for v in vecs]
        else:
            # sentence_transformers encode
            vecs = model.encode(req.texts, normalize_embeddings=True)
            embeddings = vecs.tolist()

        dim = len(embeddings[0]) if embeddings else VECTOR_SIZE
        return EmbedResponse(
            model=MODEL_NAME,
            model_type=model_type,
            dimension=dim,
            count=len(embeddings),
            embeddings=embeddings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi embed văn bản: {e}")
