"""Hỗ trợ nút In trực tiếp (Mục 8) — render HTML vào iframe ẩn rồi gọi `window.print()`."""

from __future__ import annotations

import json


def build_print_html(document_html: str) -> str:
    """Bọc HTML tài liệu vào 1 iframe + script gọi in ngay khi load — dùng với
    `st.components.v1.html(build_print_html(html), height=0)` ở Trang 5.

    `document_html` chứa nội dung do người dùng nhập/sửa (vd tên gói thầu, văn bản mục đã
    sửa) — trước đây nhét trực tiếp vào JS template literal (backtick), nên một dấu backtick
    hoặc `${...}` trong nội dung có thể THOÁT khỏi chuỗi và chèn mã JS tuỳ ý (XSS). Dùng
    `json.dumps` để mã hoá thành chuỗi JS hợp lệ (tự escape backslash/quote/backtick đúng
    cách) — an toàn hơn tự escape thủ công. Vẫn giữ chặn `</script>` để không thoát khỏi
    thẻ `<script>` bao ngoài (JSON string không tự escape `/`)."""
    payload = json.dumps(document_html).replace("</", "<\\/")
    return f"""
    <iframe id="print-frame" style="display:none;"></iframe>
    <script>
      const doc = document.getElementById('print-frame').contentWindow.document;
      const html = JSON.parse({payload});
      doc.open();
      doc.write(html);
      doc.close();
      setTimeout(() => {{
        document.getElementById('print-frame').contentWindow.focus();
        document.getElementById('print-frame').contentWindow.print();
      }}, 300);
    </script>
    """
