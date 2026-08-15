"""AutoTender-VN REST API Backend Service (FastAPI).

Expose các REST endpoints cho các ứng dụng ngoài (Postman, Mobile App, Frontend độc lập,
hoặc microservices khác) gọi trực tiếp vào hệ thống RAG & Qdrant.

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autotender.config import get_app_settings
from autotender.rag.hybrid_retriever import HybridLegalRetriever
from autotender.rag.qdrant_store import QdrantLegalStore

app = FastAPI(
    title="AutoTender-VN REST API",
    description="Hệ thống RAG + LLM tự động soạn thảo và rà soát Hồ sơ mời thầu (E-HSMT) tại Việt Nam",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy retriever singleton
_retriever: HybridLegalRetriever | None = None


def get_retriever() -> HybridLegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridLegalRetriever()
    return _retriever


# --- Pydantic Request/Response Models ---

class HealthResponse(BaseModel):
    status: str
    app_name: str
    qdrant_status: str
    qdrant_host: str
    qdrant_collection: str
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi hoặc từ khóa tìm kiếm pháp lý", example="Hồ sơ mời thầu gồm những nội dung gì?")
    top_k: int = Field(5, ge=1, le=50, description="Số lượng chunk kết quả trả về")
    rerank: bool = Field(True, description="Áp dụng cross-encoder reranking (chính xác hơn)")
    law_ids: list[str] | None = Field(None, description="Lọc theo mã văn bản luật (metadata filtering)")


class ChunkResponse(BaseModel):
    chunk_id: str
    source_doc: str
    score: float
    law_id: str | None = None
    dieu_so: int | None = None
    text: str


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[ChunkResponse]


class QARequest(BaseModel):
    question: str = Field(..., description="Câu hỏi pháp lý cần giải đáp", example="Quy định về bảo đảm dự thầu trong đấu thầu qua mạng?")
    top_k: int = Field(5, description="Số đoạn luật trích dẫn")


class QAResponse(BaseModel):
    question: str
    answer: str
    tier_used: int
    citations: list[ChunkResponse]


# --- Endpoints ---

@app.get("/", tags=["General"])
def root():
    return {
        "message": "AutoTender-VN API Backend",
        "docs": "/docs",
        "health": "/api/v1/health",
        "qdrant_dashboard": "http://localhost:6333/dashboard",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    cfg = get_app_settings()
    store = QdrantLegalStore(cfg=cfg.qdrant, vector_size=cfg.embedding.vector_size)
    qdrant_ok = store.is_available()
    return HealthResponse(
        status="healthy",
        app_name=cfg.app.name,
        qdrant_status="connected" if qdrant_ok else "disconnected",
        qdrant_host=f"{cfg.qdrant.host}:{cfg.qdrant.port}",
        qdrant_collection=cfg.qdrant.collection,
        embedding_model=cfg.embedding.model_key,
    )


@app.get("/api/v1/qdrant/collection", tags=["Vector DB"])
def get_collection_info():
    cfg = get_app_settings()
    store = QdrantLegalStore(cfg=cfg.qdrant, vector_size=cfg.embedding.vector_size)
    if not store.is_available():
        raise HTTPException(status_code=503, detail="Qdrant service is not reachable.")
    return store.collection_info()


@app.get("/api/v1/knowledge", tags=["Knowledge Base CRUD"])
def list_knowledge_documents():
    """Liệt kê toàn bộ văn bản pháp luật, số lượng Điều, số lượng Chunks và trạng thái trong Qdrant."""
    from autotender.knowledge.manager import KnowledgeManager
    km = KnowledgeManager()
    return km.list_documents()


@app.get("/api/v1/knowledge/{law_id}", tags=["Knowledge Base CRUD"])
def get_knowledge_document(law_id: str):
    """Xem chi tiết văn bản và danh sách các chunk đã phân tách theo Điều/Khoản."""
    from autotender.knowledge.manager import KnowledgeManager
    km = KnowledgeManager()
    doc = km.get_document(law_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy văn bản với law_id '{law_id}'")
    return doc


@app.delete("/api/v1/knowledge/{law_id}", tags=["Knowledge Base CRUD"])
def delete_knowledge_document(law_id: str):
    """Xóa hoàn toàn văn bản pháp luật khỏi Qdrant collection và kho lưu trữ cục bộ."""
    from autotender.knowledge.manager import KnowledgeManager
    km = KnowledgeManager()
    try:
        return km.delete_document(law_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/knowledge/{law_id}/reindex", tags=["Knowledge Base CRUD"])
def reindex_knowledge_document(law_id: str):
    """Phân tách lại (Re-chunk) và nạp lại vector của văn bản vào Qdrant."""
    from autotender.knowledge.manager import KnowledgeManager
    km = KnowledgeManager()
    try:
        return km.reindex_document(law_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search", response_model=SearchResponse, tags=["RAG Retrieval"])
def search_legal_knowledge(req: SearchRequest):
    retriever = get_retriever()
    law_set = set(req.law_ids) if req.law_ids else None
    
    if req.rerank:
        chunks = retriever.retrieve_reranked(req.query, top_k=req.top_k, law_ids=law_set)
    else:
        chunks = retriever.retrieve(req.query, top_k=req.top_k, law_ids=law_set)
        
    results = [
        ChunkResponse(
            chunk_id=c.chunk_id,
            source_doc=c.source_doc,
            score=c.score,
            law_id=c.law_id,
            dieu_so=c.dieu_so,
            text=c.text,
        )
        for c in chunks
    ]
    return SearchResponse(query=req.query, total=len(results), results=results)


@app.post("/api/v1/qa", response_model=QAResponse, tags=["Legal QA"])
def ask_legal_qa(req: QARequest):
    try:
        from autotender.models.legal_qa import LegalQAModule
        qa = LegalQAModule()
        result = qa.answer(req.question, top_k=req.top_k)
        
        citations = [
            ChunkResponse(
                chunk_id=c.chunk_id,
                source_doc=c.source_doc,
                score=c.score,
                law_id=c.law_id,
                dieu_so=c.dieu_so,
                text=c.text,
            )
            for c in result.citations
        ]
        return QAResponse(
            question=req.question,
            answer=result.answer,
            tier_used=result.tier_used,
            citations=citations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

