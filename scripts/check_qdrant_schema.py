"""Kiểm tra payload Qdrant collection — xác nhận schema mới (Named Vectors + full payload).

Script này in tóm tắt collection và lấy mẫu vài point để verify payload đầy đủ:
- chunk_id, text/content, source_doc, law_id, law_name, doc_type
- dieu_so, dieu_title, khoan_so, word_count, char_count

Ví dụ:
    python scripts/check_qdrant_schema.py
    python scripts/check_qdrant_schema.py --sample 10
    python scripts/check_qdrant_schema.py --law-id luat_22_2023_qh15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.config import get_app_settings  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

# Thống kê tổng hợp để in màu terminal (không dùng rich để giảm dependency)
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=5, help="Số point lấy mẫu để kiểm tra payload (mặc định 5)")
    parser.add_argument("--law-id", default=None, help="Lọc theo law_id cụ thể (mặc định: lấy ngẫu nhiên)")
    parser.add_argument("--json", action="store_true", help="In payload dưới dạng JSON đầy đủ thay vì tóm tắt")
    args = parser.parse_args()

    cfg = get_app_settings()

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError:
        print(_c("Lỗi: qdrant-client chưa được cài đặt.", _RED))
        sys.exit(1)

    print(_c("\n=== Kiểm tra Qdrant Collection Schema ===\n", _BOLD))

    client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, timeout=cfg.qdrant.timeout)

    # --- 1. Thông tin collection ---
    collection = cfg.qdrant.collection
    if not client.collection_exists(collection):
        print(_c(f"Collection '{collection}' KHÔNG tồn tại!", _RED))
        print("Hãy chạy: python scripts/ingest_to_qdrant.py --recreate-collection")
        sys.exit(1)

    info = client.get_collection(collection)
    vectors_cfg = info.config.params.vectors
    points_count = getattr(info, "points_count", "?")

    print(f"Collection : {_c(collection, _GREEN)}")
    print(f"Points     : {_c(str(points_count), _GREEN)}")

    if isinstance(vectors_cfg, dict):
        for vname, vparams in vectors_cfg.items():
            print(f"Vector     : {_c(f'name=\'{vname}\'', _GREEN)}, size={vparams.size}, distance={vparams.distance}")
        print(_c("✓ Named Vectors — schema đúng chuẩn", _GREEN))
    else:
        print(f"Vector     : {_c('UNNAMED (schema cũ)', _YELLOW)} size={vectors_cfg.size}")
        print(_c("⚠ Chưa dùng Named Vectors — hãy chạy: ingest_to_qdrant.py --recreate-collection", _YELLOW))

    # Payload index
    payload_schema = getattr(info, "payload_schema", {}) or {}
    if payload_schema:
        print(f"Payload index: {', '.join(payload_schema.keys())}")
    print()

    # --- 2. Thống kê theo doc_type ---
    print(_c("── Thống kê theo doc_type ──", _BOLD))
    for doc_type in ["Luật", "Nghị định", "Thông tư", "Khác"]:
        try:
            result = client.count(
                collection_name=collection,
                count_filter=Filter(must=[
                    FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
                ]),
            )
            count = result.count
            bar = "█" * min(count // 5, 40)
            print(f"  {doc_type:<12} {count:>4} points  {bar}")
        except Exception as e:  # noqa: BLE001
            print(f"  {doc_type:<12} (lỗi: {e})")

    # Thống kê theo law_id
    print(_c("\n── Thống kê theo law_id ──", _BOLD))
    law_ids = [
        "luat_22_2023_qh15",
        "nd_214_2025_ndcp",
        "nd_45_2026_ndcp",
        "tt_01_2024_bkhdt",
        "tt_22_2024_bkhdt",
    ]
    for lid in law_ids:
        try:
            result = client.count(
                collection_name=collection,
                count_filter=Filter(must=[
                    FieldCondition(key="law_id", match=MatchValue(value=lid))
                ]),
            )
            count = result.count
            icon = _c("✓", _GREEN) if count > 0 else _c("✗", _RED)
            print(f"  {icon} {lid:<30} {count:>4} points")
        except Exception as e:  # noqa: BLE001
            print(f"  {_c('!', _RED)} {lid:<30} (lỗi: {e})")

    # --- 3. Lấy mẫu points ---
    print(_c(f"\n── Mẫu {args.sample} points ──", _BOLD))
    scroll_filter = None
    if args.law_id:
        scroll_filter = Filter(must=[FieldCondition(key="law_id", match=MatchValue(value=args.law_id))])

    result, _ = client.scroll(
        collection_name=collection,
        with_payload=True,
        with_vectors=False,
        limit=args.sample,
        scroll_filter=scroll_filter,
    )

    required_fields = {"chunk_id", "text", "content", "source_doc", "law_id", "law_name",
                       "doc_type", "dieu_so", "word_count"}

    for i, point in enumerate(result, 1):
        p = point.payload or {}
        if args.json:
            print(f"\n[Point {i}] id={point.id}")
            print(json.dumps(p, ensure_ascii=False, indent=2))
            continue

        print(f"\n{_c(f'[Point {i}]', _BOLD)} id={point.id}")
        print(f"  chunk_id   : {p.get('chunk_id', _c('MISSING', _RED))}")
        print(f"  doc_type   : {_c(p.get('doc_type', '?'), _GREEN)}")
        print(f"  law_id     : {p.get('law_id', _c('MISSING', _RED))}")
        print(f"  law_name   : {p.get('law_name', _c('MISSING', _RED))}")
        print(f"  dieu_so    : {p.get('dieu_so', _c('MISSING', _RED))}")
        dieu_title = p.get('dieu_title')
        print(f"  dieu_title : {_c(dieu_title, _GREEN) if dieu_title else _c('None (chunk trọn Điều = OK)', _YELLOW)}")
        khoan_so = p.get('khoan_so')
        print(f"  khoan_so   : {khoan_so if khoan_so else _c('None (trọn Điều)', _YELLOW)}")
        print(f"  word_count : {p.get('word_count', _c('MISSING', _RED))}")
        print(f"  char_count : {p.get('char_count', _c('MISSING', _RED))}")
        content = p.get('content') or p.get('text', '')
        print(f"  content    : {content[:100].strip()}..." if len(content) > 100 else f"  content: {content}")

        # Kiểm tra completeness
        missing = required_fields - set(p.keys())
        if missing:
            print(f"  {_c('⚠ THIẾU FIELDS:', _YELLOW)} {', '.join(sorted(missing))}")
        else:
            print(f"  {_c('✓ Payload đầy đủ', _GREEN)}")

    # Dashboard link
    dashboard = f"http://{cfg.qdrant.host}:{cfg.qdrant.port}/dashboard"
    print(_c(f"\n── Dashboard ──", _BOLD))
    print(f"  {dashboard}#/collections/{collection}")
    print(f"  → Visualize: chọn vector 'dense', color_by: doc_type hoặc law_id")
    print(f"  → Filter: bấm 'Scroll' → thêm filter doc_type='Nghị định'")


if __name__ == "__main__":
    main()
