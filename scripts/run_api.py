"""Khởi chạy AutoTender-VN REST API Backend Server (FastAPI / Uvicorn).

Cách dùng:
    python scripts/run_api.py
    python scripts/run_api.py --port 8000 --reload

Sau khi chạy:
    - Swagger UI tài liệu API: http://localhost:8000/docs
    - ReDoc tài liệu API:    http://localhost:8000/redoc
    - Health check endpoint:  http://localhost:8000/api/v1/health
"""

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="Host IP để bind server (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port để chạy API server (mặc định: 8000)")
    parser.add_argument("--reload", action="store_true", help="Tự động reload server khi sửa code")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Khởi chạy AutoTender-VN REST API Server")
    print(f"📡 API Endpoints: http://localhost:{args.port}")
    print(f"📖 Swagger Docs:  http://localhost:{args.port}/docs")
    print(f"🔍 Health check:  http://localhost:{args.port}/api/v1/health")
    print("=" * 60)

    uvicorn.run(
        "autotender.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
