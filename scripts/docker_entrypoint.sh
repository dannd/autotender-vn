#!/bin/bash
# docker_entrypoint.sh — entrypoint cho AutoTender-VN app container.
#
# Chờ Qdrant sẵn sàng, ingest kho tri thức nếu collection chưa có dữ liệu,
# rồi start Streamlit.
#
# Biến môi trường:
#   QDRANT_HOST  (mặc định: qdrant — service name trong docker-compose.yml)
#   QDRANT_PORT  (mặc định: 6333)
#   ANTHROPIC_API_KEY (tuỳ chọn — nếu không có, app chạy ở Degraded Mode Tier 3)

set -e

QDRANT_HOST="${QDRANT_HOST:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
MAX_WAIT=60  # giây chờ Qdrant ready tối đa

echo "[entrypoint] Chờ Qdrant tại ${QDRANT_HOST}:${QDRANT_PORT}..."
waited=0
until curl -sf "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" > /dev/null 2>&1; do
    if [ $waited -ge $MAX_WAIT ]; then
        echo "[entrypoint] CẢNH BÁO: Qdrant chưa ready sau ${MAX_WAIT}s — app vẫn khởi động ở Degraded Mode (BM25-only)."
        break
    fi
    echo "[entrypoint] Qdrant chưa ready, thử lại sau 2s... (${waited}s/${MAX_WAIT}s)"
    sleep 2
    waited=$((waited + 2))
done

echo "[entrypoint] Kiểm tra và ingest kho tri thức vào Qdrant (idempotent)..."
python scripts/ingest_to_qdrant.py --dry-run
# Nếu collection chưa có dữ liệu, ingest đầy đủ
COLLECTION_COUNT=$(python -c "
from autotender.config import get_app_settings
from autotender.rag.qdrant_store import QdrantLegalStore
import sys
sys.path.insert(0, 'src')
cfg = get_app_settings()
store = QdrantLegalStore(cfg=cfg.qdrant, vector_size=cfg.embedding.vector_size)
info = store.collection_info()
print(info.get('vectors_count', 0))
" 2>/dev/null || echo "0")

if [ "${COLLECTION_COUNT}" -eq "0" ] 2>/dev/null; then
    echo "[entrypoint] Collection rỗng — đang ingest kho tri thức..."
    python scripts/ingest_to_qdrant.py
    echo "[entrypoint] Ingest hoàn tất."
else
    echo "[entrypoint] Collection đã có ${COLLECTION_COUNT} vectors — bỏ qua ingest."
fi

echo "[entrypoint] Khởi động Streamlit..."
exec streamlit run app/main.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true
