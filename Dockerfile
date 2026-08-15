# AutoTender-VN — image chạy ứng dụng Streamlit (v2, Qdrant backend).
#
# Thay đổi so với v1:
# - Bỏ `RUN python scripts/build_legal_index.py` (FAISS) — index nay lưu trong Qdrant
#   (persistent volume), không build vào image.
# - Thêm qdrant-client dependency.
# - Dùng scripts/docker_entrypoint.sh làm CMD — tự ingest vào Qdrant lần đầu rồi mới
#   start Streamlit.
#
# Không cài paddleocr/paddlepaddle và playwright (xem README.md cho lý do).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Cài curl (cần cho health check Qdrant trong docker_entrypoint.sh)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Torch từ index CPU-only của PyTorch — tránh pull build CUDA nặng thêm vài GB.
RUN grep -v -E '^(paddleocr|paddlepaddle)' requirements.txt > /tmp/requirements-docker.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-docker.txt

COPY . .

# Đảm bảo entrypoint script có quyền thực thi
RUN chmod +x scripts/docker_entrypoint.sh

# Corpus luật thật đã có sẵn trong repo (data/samples/legal_corpus/*.jsonl).
# Ingest vào Qdrant sẽ thực hiện lúc container startup qua docker_entrypoint.sh
# (không phải lúc build image) — container cần network tới Qdrant để ingest.

# Sau khi container chạy, tạo tài khoản đầu tiên:
#   docker exec -it <container> python scripts/create_user.py --username admin --role admin
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["scripts/docker_entrypoint.sh"]

