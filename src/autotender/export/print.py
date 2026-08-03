"""Hỗ trợ nút In trực tiếp (Mục 8) — render HTML vào iframe ẩn rồi gọi `window.print()`."""

from __future__ import annotations


def build_print_html(document_html: str) -> str:
    """Bọc HTML tài liệu vào 1 iframe + script gọi in ngay khi load — dùng với
    `st.components.v1.html(build_print_html(html), height=0)` ở Trang 5."""
    escaped = document_html.replace("</script>", "<\\/script>")
    return f"""
    <iframe id="print-frame" style="display:none;"></iframe>
    <script>
      const doc = document.getElementById('print-frame').contentWindow.document;
      doc.open();
      doc.write(`{escaped}`);
      doc.close();
      setTimeout(() => {{
        document.getElementById('print-frame').contentWindow.focus();
        document.getElementById('print-frame').contentWindow.print();
      }}, 300);
    </script>
    """
