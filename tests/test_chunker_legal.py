from datetime import datetime, timezone

from autotender.rag.chunker import MAX_WORDS, chunk_legal_article
from autotender.schemas import LegalArticle


def _article(text: str, dieu_so: int = 1, dieu_title: str = "Phạm vi điều chỉnh") -> LegalArticle:
    return LegalArticle(
        law_id="luat_gia_lap",
        law_name="Luật giả lập",
        chuong_so="I",
        chuong_title="QUY ĐỊNH CHUNG",
        dieu_so=dieu_so,
        dieu_title=dieu_title,
        text=text,
        source_url="https://example.com",
        fetched_at=datetime.now(timezone.utc),
    )


def test_short_article_becomes_single_chunk():
    article = _article("1. Nội dung ngắn gọn của điều này.")
    chunks = chunk_legal_article(article)

    assert len(chunks) == 1
    assert chunks[0].law_id == "luat_gia_lap"
    assert chunks[0].dieu_so == 1
    assert chunks[0].source_doc == "Luật giả lập — Điều 1. Phạm vi điều chỉnh"
    assert chunks[0].text == article.text


def test_long_article_splits_at_khoan_boundaries_not_mid_sentence():
    khoan_text = " ".join(f"từ{i}" for i in range(120))
    text = "\n\n".join(f"{k}. {khoan_text}" for k in range(1, 6))  # 5 khoản, ~600 từ tổng

    article = _article(text)
    chunks = chunk_legal_article(article)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= MAX_WORDS
        # mỗi chunk phải bắt đầu bằng một khoản trọn vẹn (số + dấu chấm), không cắt giữa câu
        assert c.text.strip()[0].isdigit()
        assert "Khoản" in c.source_doc


def test_khoan_metadata_preserved_across_chunks():
    khoan_text = " ".join(f"từ{i}" for i in range(150))
    text = "\n\n".join(f"{k}. {khoan_text}" for k in range(1, 4))
    article = _article(text, dieu_so=44, dieu_title="Nội dung hồ sơ mời thầu")

    chunks = chunk_legal_article(article)
    for c in chunks:
        assert c.dieu_so == 44
        assert c.law_id == "luat_gia_lap"
