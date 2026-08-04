from autotender.knowledge.legal_fetch import extract_body, parse_articles

_FIXTURE_FULL_PAGE = """
BÁO ĐIỆN TỬ CHÍNH PHỦ
Một số menu điều hướng không liên quan
(Chinhphu.vn) - Toàn văn Luật giả lập.
LUẬT
GIẢ LẬP
 Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh

Nội dung điều 1 dòng một.
Nội dung điều 1 dòng hai.

Điều 2. Giải thích từ ngữ

Nội dung điều 2.

Chương II
TIÊU CHUẨN ĐÁNH GIÁ
Điều 3. Nội dung hồ sơ mời thầu

1. Hồ sơ mời thầu bao gồm:
a) Chỉ dẫn nhà thầu;
b) Bảng dữ liệu đấu thầu.

________________

Luật này được Quốc hội thông qua ngày 1 tháng 1 năm 2024.

Tham khảo thêm
Bài viết liên quan không thuộc luật
"""


def test_extract_body_slices_between_markers():
    body = extract_body(_FIXTURE_FULL_PAGE, "Chương I", "Tham khảo thêm")
    assert body.startswith("Chương I")
    assert "Tham khảo thêm" not in body
    assert "BÁO ĐIỆN TỬ CHÍNH PHỦ" not in body


def test_extract_body_raises_on_missing_marker():
    try:
        extract_body(_FIXTURE_FULL_PAGE, "Không tồn tại", "Tham khảo thêm")
        assert False, "phải raise ValueError"
    except ValueError:
        pass


def test_parse_articles_groups_by_chapter_and_extracts_verbatim_text():
    body = extract_body(_FIXTURE_FULL_PAGE, "Chương I", "Tham khảo thêm")
    articles = parse_articles(body, law_id="luat_gia_lap", law_name="Luật giả lập", source_url="https://example.com")

    assert len(articles) == 3
    a1, a2, a3 = articles

    assert a1.dieu_so == 1
    assert a1.dieu_title == "Phạm vi điều chỉnh"
    assert a1.chuong_so == "I"
    assert a1.chuong_title == "QUY ĐỊNH CHUNG"
    assert "Nội dung điều 1 dòng một." in a1.text
    assert "Nội dung điều 1 dòng hai." in a1.text
    assert "Điều 2" not in a1.text  # không lẫn sang điều sau

    assert a2.dieu_so == 2
    assert a2.chuong_so == "I"

    assert a3.dieu_so == 3
    assert a3.chuong_so == "II"
    assert a3.chuong_title == "TIÊU CHUẨN ĐÁNH GIÁ"
    assert "Chỉ dẫn nhà thầu" in a3.text


def test_parse_articles_uses_initial_chuong_when_first_chapter_has_no_roman_numeral():
    """Một số văn bản (vd Nghị định 24/2024) không render 'Chương I' — chỉ có tiêu đề."""
    fixture = (
        "NHỮNG QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi điều chỉnh\n\n"
        "Nội dung điều 1.\n\n"
        "Chương II\n"
        "ĐIỀU KHOẢN KHÁC\n"
        "Điều 2. Nội dung khác\n\n"
        "Nội dung điều 2.\n"
    )
    articles = parse_articles(
        fixture, law_id="x", law_name="x", source_url="https://x",
        initial_chuong_so="I", initial_chuong_title="NHỮNG QUY ĐỊNH CHUNG",
    )
    assert articles[0].chuong_so == "I"
    assert articles[0].chuong_title == "NHỮNG QUY ĐỊNH CHUNG"
    assert articles[1].chuong_so == "II"


def test_parse_articles_raises_when_no_articles_found():
    try:
        parse_articles("không có điều nào ở đây", law_id="x", law_name="x", source_url="https://x")
        assert False, "phải raise ValueError"
    except ValueError:
        pass
