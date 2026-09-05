"""FastAPI Server phục vụ mô hình nhúng deepx-embedding-v1 (1024 chiều).

Hỗ trợ chạy Native trên máy host (macOS M-series / CUDA / CPU) hoặc trong container Docker.
- Port: 8080
- Model: dxtech-asia/deepx-embedding-v1 (1024d, 8K context, Linear Attention)
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("embedding-service")

# Tối ưu CPU multi-threading
num_threads = min(os.cpu_count() or 8, 14)
try:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    logger.info("Torch configured with %d CPU threads", num_threads)
except Exception:
    pass

MODEL_NAME = os.getenv("MODEL_NAME", "dxtech-asia/deepx-embedding-v1")
MODEL_KEY = os.getenv("MODEL_KEY", "deepx_v1")
DEFAULT_DIM = int(os.getenv("VECTOR_SIZE", "1024"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="DeepX Embedding Service",
    description="Microservice phục vụ vector hóa văn bản pháp lý (deepx-embedding-v1, 1024 chiều)",
    version="1.0.0",
)

_model = None
_tokenizer = None
_id_remap = None
_model_type = None


def _init_deepx():
    """Khởi tạo mô hình DeepX Embedding v1.0."""
    global _model, _tokenizer, _id_remap, _model_type

    logger.info("Đang khởi tạo DeepX model '%s' trên device '%s'...", MODEL_NAME, DEVICE)

    # 1. Khắc phục xung đột tên module 'config' nếu có
    try:
        from config import DeepXConfig
        from modeling.pipeline import DeepXPipeline
    except (ImportError, AttributeError):
        site_packages = [p for p in sys.path if "site-packages" in p]
        for sp in site_packages:
            cfg_file = Path(sp) / "config.py"
            if cfg_file.exists():
                spec = importlib.util.spec_from_file_location("config", str(cfg_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["config"] = mod
                    spec.loader.exec_module(mod)
                    break
        from config import DeepXConfig
        from modeling.pipeline import DeepXPipeline

    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    # 2. Tải tokenizer & weights từ HuggingFace Hub
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    ckpt_path = Path(hf_hub_download(MODEL_NAME, "deepx_v1.0.pt"))
    embed_path = Path(hf_hub_download(MODEL_NAME, "token_embedding.pt"))
    remap_path = Path(hf_hub_download(MODEL_NAME, "id_remap.pt"))

    id_remap = torch.load(remap_path, map_location="cpu") if remap_path.exists() else None

    # 3. Khởi tạo Pipeline & nạp weights
    deepx_config = DeepXConfig()
    pipeline = DeepXPipeline(deepx_config)

    embed_data = torch.load(embed_path, map_location="cpu", weights_only=True)
    embed_weight = embed_data["weight"] if isinstance(embed_data, dict) and "weight" in embed_data else embed_data
    pipeline.token_embedding = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1])
    pipeline.token_embedding.weight = nn.Parameter(embed_weight.float(), requires_grad=False)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    pipeline.backbone.load_state_dict(sd, strict=False)
    pipeline = pipeline.float().eval()

    for m in pipeline.modules():
        if hasattr(m, "path_mix_logit"):
            m._alpha_override = 0.0

    pipeline.token_embedding = pipeline.token_embedding.to("cpu")
    pipeline.backbone = pipeline.backbone.to(DEVICE)

    _model = pipeline
    _tokenizer = tokenizer
    _id_remap = id_remap
    _model_type = "deepx_pipeline"
    logger.info("DeepX Model loaded successfully on device: %s!", DEVICE)


def get_model():
    """Lazy load model khi có request đầu tiên."""
    global _model, _tokenizer, _id_remap, _model_type
    if _model is None:
        try:
            _init_deepx()
        except Exception as e:
            logger.warning("Không thể load DeepX pipeline (%s). Fallback sang SentenceTransformer...", e)
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            _model_type = "sentence_transformers"
            logger.info("Loaded fallback SentenceTransformer: %s", MODEL_NAME)
    return _model, _tokenizer, _id_remap, _model_type


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="Danh sách các đoạn văn bản cần vector hóa")
    batch_size: int = Field(16, description="Kích thước batch")
    dimension: int | None = Field(None, description="Chiều vector mong muốn (Matryoshka: 256, 512, 768, 1024, 1536)")


class EmbedResponse(BaseModel):
    model: str
    model_type: str
    dimension: int
    dimensions: int
    count: int
    embeddings: list[list[float]]


@app.get("/")
def root():
    return {
        "service": "AutoTender DeepX Embedding Service",
        "model": MODEL_NAME,
        "device": DEVICE,
        "endpoints": ["/health", "/info", "/embed"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_key": MODEL_KEY,
        "device": DEVICE,
    }


@app.get("/info")
def info():
    return {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "default_vector_size": DEFAULT_DIM,
        "context_length": 8192,
        "architecture": "Linear Attention (Gated DeltaNet-2, O(n))",
        "device": DEVICE,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        return EmbedResponse(
            model=MODEL_NAME,
            model_type=_model_type or "none",
            dimension=DEFAULT_DIM,
            dimensions=DEFAULT_DIM,
            count=0,
            embeddings=[],
        )

    model, tokenizer, id_remap, model_type = get_model()
    target_dim = req.dimension or DEFAULT_DIM
    batch_size = max(1, req.batch_size)
    all_embeddings: list[list[float]] = []

    try:
        if model_type == "deepx_pipeline":
            for i in range(0, len(req.texts), batch_size):
                batch_texts = req.texts[i : i + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                mask = encoded["attention_mask"].to(DEVICE)

                if id_remap is not None:
                    input_ids = id_remap[input_ids]

                max_actual = int(mask.sum(dim=1).max().item())
                input_ids = input_ids[:, :max_actual]
                mask = mask[:, :max_actual]

                hidden = model.token_embedding(input_ids).to(DEVICE).float()
                with torch.no_grad():
                    emb = model.backbone(hidden, attention_mask=mask, normalize=False)
                    if target_dim is not None and target_dim < emb.shape[1]:
                        emb = emb[:, :target_dim]
                    emb = F.normalize(emb, p=2, dim=-1)
                all_embeddings.extend(emb.cpu().tolist())
        else:
            # Fallback SentenceTransformer
            vecs = model.encode(req.texts, normalize_embeddings=True)
            if isinstance(vecs, np.ndarray):
                all_embeddings = vecs.tolist()
            else:
                all_embeddings = [v.tolist() for v in vecs]

        dim = len(all_embeddings[0]) if all_embeddings else target_dim
        return EmbedResponse(
            model=MODEL_NAME,
            model_type=model_type,
            dimension=dim,
            dimensions=dim,
            count=len(all_embeddings),
            embeddings=all_embeddings,
        )
    except Exception as e:
        logger.error("Lỗi khi embed văn bản: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi embed văn bản: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
