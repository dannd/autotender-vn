# AutoTender-VN — image chạy ứng dụng Streamlit cho bản beta.
#
# Không cài paddleocr/paddlepaddle (OCR tuỳ chọn, xem src/autotender/ingest/ocr.py — hệ
# thống tự bỏ qua nếu thiếu) và không cài playwright/chromium (chỉ cần cho
# scripts/fetch_legal_corpus.py chạy 1 lần lúc dựng kho tri thức, KHÔNG cần lúc app chạy)
# — giữ image nhỏ và build nhanh, đúng những gì app thật sự cần lúc runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
# Torch từ index CPU-only của PyTorch trước — tránh pip kéo build có CUDA (nặng thêm vài
# GB) từ PyPI mặc định cho 1 image không có GPU.
RUN grep -v -E '^(paddleocr|paddlepaddle)' requirements.txt > /tmp/requirements-docker.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-docker.txt

COPY . .

# Dựng sẵn FAISS index cho model embedding mặc định (vi_bi_encoder, khớp cấu hình
# configs/models.yaml) ngay lúc build image — mọi container khởi động từ cùng 1 image có
# index giống hệt nhau, không phụ thuộc mạng lúc runtime. Nguồn (data/samples/legal_corpus/)
# đã có sẵn trong repo, không cần fetch lại. Muốn build đủ 3 model để dùng Trang 8 — Đánh
# giá (bảng so sánh embedding) thì bỏ tham số --model.
RUN python scripts/build_legal_index.py --model vi_bi_encoder

# Sau khi build image, PHẢI tạo ít nhất 1 tài khoản trước khi đăng nhập được (không có tài
# khoản mặc định — xem app/auth_ui.py):
#   docker exec -it <container> python scripts/create_user.py --username admin --role admin
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
