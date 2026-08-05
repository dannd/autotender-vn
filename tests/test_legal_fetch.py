from autotender.knowledge.legal_fetch import extract_body, parse_articles, parse_gxd_theme_articles

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


def _gxd_heading(tag: str, id_: str, text: str) -> str:
    return (
        f'<{tag} id="{id_}" class="gxd-copy-target">'
        f'<a class="header-anchor" href="#{id_}"><span>{text}</span></a>'
        f'<div class="gxd-paragraph-tools"><button class="gxd-paragraph-tools-item">Copy đoạn</button>'
        f'<button class="gxd-paragraph-tools-item">Ghi chú</button></div>'
        f"</{tag}>"
    )


def _gxd_paragraph(text: str) -> str:
    return f'<p class="gxd-copy-target">{text}</p>'


_GXD_FIXTURE_PAGE = (
    '<div class="theme-default-content"><div>'
    + _gxd_heading("h3", "dieu-1-pham-vi", "Điều 1. Phạm vi điều chỉnh")
    + _gxd_paragraph("Nội dung điều 1.")
    + _gxd_heading("h2", "chuong-ii-x", "Chương II TIÊU CHUẨN ĐÁNH GIÁ")
    + _gxd_heading("h3", "dieu-2-x", "Điều 2. Nội dung trùng lặp")
    + _gxd_paragraph("Bản đầu, ngắn.")
    + _gxd_heading("h3", "dieu-2-x", "Điều 2. Nội dung trùng lặp")
    + _gxd_paragraph("Bản hai, khác nội dung.")
    + _gxd_heading("h3", "dieu-3-x", "Điều 3. Điều bình thường")
    + _gxd_paragraph("Nội dung điều 3 hợp lệ.")
    + "</div></div>"
)


def test_parse_gxd_theme_articles_extracts_clean_text_ignoring_toolbar_noise():
    articles = parse_gxd_theme_articles(
        _GXD_FIXTURE_PAGE, law_id="x", law_name="x", source_url="https://x",
        initial_chuong_so="I", initial_chuong_title="QUY ĐỊNH CHUNG",
    )
    a1 = next(a for a in articles if a.dieu_so == 1)
    assert a1.chuong_so == "I"
    assert a1.text == "Nội dung điều 1."
    assert "Copy đoạn" not in a1.text
    assert "Ghi chú" not in a1.text


def test_parse_gxd_theme_articles_drops_article_with_duplicate_heading():
    articles = parse_gxd_theme_articles(
        _GXD_FIXTURE_PAGE, law_id="x", law_name="x", source_url="https://x",
        initial_chuong_so="I", initial_chuong_title="QUY ĐỊNH CHUNG",
    )
    nums = [a.dieu_so for a in articles]
    assert 2 not in nums  # heading lặp -> loại bỏ hoàn toàn, không giữ bản nào
    assert 1 in nums
    assert 3 in nums


def test_parse_gxd_theme_articles_uses_chuong_ii_for_dieu_3():
    articles = parse_gxd_theme_articles(
        _GXD_FIXTURE_PAGE, law_id="x", law_name="x", source_url="https://x",
        initial_chuong_so="I", initial_chuong_title="QUY ĐỊNH CHUNG",
    )
    a3 = next(a for a in articles if a.dieu_so == 3)
    assert a3.chuong_so == "II"
    assert a3.chuong_title == "TIÊU CHUẨN ĐÁNH GIÁ"


def test_parse_gxd_theme_articles_drops_oversized_merged_article():
    huge_text = " ".join(f"từ{i}" for i in range(2500))
    fixture = (
        '<div class="theme-default-content"><div>'
        + _gxd_heading("h3", "dieu-1-x", "Điều 1. Điều bị trộn nội dung")
        + _gxd_paragraph(huge_text)
        + _gxd_heading("h3", "dieu-2-x", "Điều 2. Điều bình thường")
        + _gxd_paragraph("Nội dung ngắn gọn.")
        + "</div></div>"
    )
    articles = parse_gxd_theme_articles(fixture, law_id="x", law_name="x", source_url="https://x")
    nums = [a.dieu_so for a in articles]
    assert 1 not in nums  # dài bất thường -> loại bỏ
    assert 2 in nums


def test_parse_gxd_theme_articles_raises_when_no_articles_found():
    fixture = '<div class="theme-default-content"><div><p>không có điều nào</p></div></div>'
    try:
        parse_gxd_theme_articles(fixture, law_id="x", law_name="x", source_url="https://x")
        assert False, "phải raise ValueError"
    except ValueError:
        pass
