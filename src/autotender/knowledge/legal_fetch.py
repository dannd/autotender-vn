"""Fetch + parse NGUYÊN VĂN văn bản pháp luật thật từ nguồn công khai chính thống
(báo điện tử Chính phủ, Công báo...) thành `LegalArticle` theo từng Điều.

Quan trọng: dùng Playwright (render JS) thay vì httpx thuần — các trang này là SPA,
nội dung không có trong HTML thô. KHÔNG dùng công cụ WebFetch (tóm tắt qua model nhỏ)
vì cần giữ nguyên văn tuyệt đối, không diễn giải — bắt buộc cho một hệ RAG pháp lý.

Đã kiểm tra robots.txt trước khi chọn nguồn: KHÔNG dùng vbpl.vn (chặn đúng path
`/Pages/` cần dùng) và KHÔNG dùng thuvienphapluat.vn (robots.txt chặn tường minh
`User-agent: ClaudeBot` — tôn trọng ý muốn của chủ site dù có thể đặt User-Agent khác
để né). Dùng xaydungchinhsach.chinhphu.vn (Báo điện tử Chính phủ) — robots.txt cho
phép toàn bộ, không có bot-specific disallow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from autotender.schemas import LegalArticle
from autotender.utils.logging import get_logger

logger = get_logger(__name__)

_CHUONG_RE = re.compile(r"^Chương\s+([IVXLCDM]+)\.?\s*(.*)$")
_DIEU_RE = re.compile(r"^Điều\s+(\d+)\.\s*(.+)$")


@dataclass
class LegalDocSource:
    law_id: str
    law_name: str
    url: str
    start_marker: str  # chuỗi đánh dấu điểm BẮT ĐẦU nội dung thật (thường là "Chương I")
    end_marker: str  # chuỗi đánh dấu điểm KẾT THÚC (trước phần "Tham khảo thêm"/tin liên quan)
    # Tuỳ chọn: một số văn bản không render số La Mã cho chương đầu (xem parse_articles).
    initial_chuong_so: str | None = None
    initial_chuong_title: str | None = None
    # Tuỳ chọn: lần xuất hiện thứ mấy của start_marker là nội dung thật (2 nếu có Mục lục
    # lặp lại heading — xem extract_body).
    start_occurrence: int = 1


def fetch_rendered_text(url: str, timeout_ms: int = 30000) -> str:
    """Render trang bằng Playwright headless, trả về toàn bộ text hiển thị của <body>."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            text = page.inner_text("body")
        finally:
            browser.close()
    return text


def extract_body(full_text: str, start_marker: str, end_marker: str, start_occurrence: int = 1) -> str:
    """Cắt phần thân văn bản luật thật ra khỏi nav/footer/tin liên quan của trang báo.

    `start_occurrence`: một số trang (vd văn bản hợp nhất) có Mục lục lặp lại đúng heading
    "Chương I ..." trước phần nội dung thật — truyền 2 để lấy lần xuất hiện thứ 2 (nội
    dung thật), bỏ qua Mục lục.
    """
    start_idx = -1
    search_from = 0
    for _ in range(start_occurrence):
        start_idx = full_text.find(start_marker, search_from)
        if start_idx < 0:
            break
        search_from = start_idx + 1
    if start_idx < 0:
        raise ValueError(
            f"Không tìm thấy start_marker: {start_marker!r} (lần xuất hiện thứ {start_occurrence}) "
            "— trang có thể đã đổi cấu trúc."
        )
    end_idx = full_text.find(end_marker, start_idx)
    if end_idx < 0:
        raise ValueError(f"Không tìm thấy end_marker: {end_marker!r} — trang có thể đã đổi cấu trúc.")
    return full_text[start_idx:end_idx]


_FOOTNOTE_MARKER_RE = re.compile(r"\[\d+\]")


def strip_footnote_markers(text: str) -> str:
    """Loại bỏ chú thích dạng `[157]` chen giữa số Điều/Khoản và nội dung — hay gặp ở văn
    bản hợp nhất (vd `Điều 90[157]. Tiêu đề`), nếu không sẽ làm regex khớp Điều/Khoản sai.
    Không mất thông tin pháp lý vì đây chỉ là số thứ tự chú thích, không phải nội dung.
    """
    return _FOOTNOTE_MARKER_RE.sub("", text)


def _is_meaningful_text(text: str) -> bool:
    """`⋮` là ký tự phân cách trang trí giữa các khoản trên một số trang (vd văn bản hợp
    nhất) — loại các dòng chỉ chứa ký tự này để không hiểu nhầm là có nội dung."""
    stripped = text.replace("⋮", "").strip()
    return len(stripped) >= 15


def parse_articles(
    body_text: str,
    law_id: str,
    law_name: str,
    source_url: str,
    initial_chuong_so: str | None = None,
    initial_chuong_title: str | None = None,
) -> list[LegalArticle]:
    """Parse thân văn bản thành list[LegalArticle], mỗi Điều 1 bản ghi kèm Chương chứa nó.

    Dùng cách quét theo dòng (không phải regex toàn văn) vì heading "Chương X" và tiêu đề
    chương nằm trên 2 dòng riêng, và số điều/khoản lồng nhau — quét dòng chắc chắn hơn.

    `initial_chuong_so`/`initial_chuong_title`: một số văn bản (vd Nghị định 24/2024) không
    render dòng "Chương I" cho chương đầu tiên (chỉ có tiêu đề, thiếu số La Mã) — truyền tay
    2 tham số này để các Điều đầu vẫn được gắn đúng chương thay vì để trống.
    """
    lines = body_text.splitlines()
    articles: list[LegalArticle] = []

    current_chuong_so: str | None = initial_chuong_so
    current_chuong_title: str | None = initial_chuong_title
    current_dieu_so: int | None = None
    current_dieu_title: str | None = None
    current_lines: list[str] = []
    now = datetime.now(timezone.utc)

    def flush() -> None:
        if current_dieu_so is not None:
            # Loại dòng chỉ chứa "⋮" (phân cách trang trí giữa các khoản, không phải nội dung).
            meaningful_lines = [ln for ln in current_lines if ln.strip() != "⋮"]
            text = "\n".join(meaningful_lines).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)  # gộp dòng trống thừa sau khi bỏ "⋮"
            if not _is_meaningful_text(text):
                logger.warning(
                    "Điều %d (%s) không có nội dung thật (có thể bị ẩn/collapsed trên trang nguồn) — bỏ qua.",
                    current_dieu_so, law_id,
                )
                return
            articles.append(
                LegalArticle(
                    law_id=law_id,
                    law_name=law_name,
                    chuong_so=current_chuong_so,
                    chuong_title=current_chuong_title,
                    dieu_so=current_dieu_so,
                    dieu_title=current_dieu_title or "",
                    text=text,
                    source_url=source_url,
                    fetched_at=now,
                )
            )

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        chuong_match = _CHUONG_RE.match(line)
        if chuong_match:
            flush()
            current_dieu_so, current_dieu_title, current_lines = None, None, []
            current_chuong_so = chuong_match.group(1)
            same_line_title = chuong_match.group(2).strip()
            if same_line_title:
                # Định dạng "Chương I QUY ĐỊNH CHUNG" trên cùng 1 dòng (vd văn bản hợp nhất)
                current_chuong_title = same_line_title
                i += 1
            else:
                # Định dạng "Chương I" và tiêu đề nằm ở dòng kế tiếp (vd trang gốc)
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                current_chuong_title = lines[j].strip() if j < len(lines) else None
                i = j + 1
            continue

        dieu_match = _DIEU_RE.match(line)
        if dieu_match:
            flush()
            current_dieu_so = int(dieu_match.group(1))
            current_dieu_title = dieu_match.group(2).strip()
            current_lines = []
            i += 1
            continue

        if current_dieu_so is not None:
            current_lines.append(lines[i])
        i += 1

    flush()

    if not articles:
        raise ValueError(f"Không parse được Điều nào từ {law_id} — kiểm tra lại pattern/marker.")
    return articles


def fetch_and_parse(source: LegalDocSource) -> list[LegalArticle]:
    logger.info("Đang fetch %s (%s)...", source.law_name, source.url)
    full_text = fetch_rendered_text(source.url)
    body = extract_body(full_text, source.start_marker, source.end_marker, start_occurrence=source.start_occurrence)
    body = strip_footnote_markers(body)
    articles = parse_articles(
        body, source.law_id, source.law_name, source.url,
        initial_chuong_so=source.initial_chuong_so, initial_chuong_title=source.initial_chuong_title,
    )
    logger.info("Parse được %d Điều từ %s.", len(articles), source.law_name)
    return articles
