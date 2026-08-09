from autotender.crawler.parser import (
    enrich_dauthau_asia_detail,
    enrich_dauthau_asia_khlcnt_detail,
    extract_khlcnt_task_summary,
    parse_dauthau_asia_khlcnt_rows,
    parse_dauthau_asia_rows,
)
from autotender.schemas import TenderNotice

_SAMPLE_HTML = """
<table class="bidding-table">
<tr>
    <td class="order-header" data-column="Gói thầu">
        <div class="wrap__text">
            <a title="Chi phí xây dựng" href="/thongbao/moithau/chi-phi-xay-dung-1785812.html"><span class="bidding-code">IB2600417491-00</span> Chi phí xây dựng</a>
        </div>
    </td>
    <td data-column="Chủ đầu tư">
        <div>
            <a title="Phòng Kinh tế xã Nhân Cơ" href="/project-owner/phong-kinh-te-xa-nhan-co-191379/">  <span class="solicitor-code">vnz000040623</span>  Phòng Kinh tế xã Nhân Cơ
            </a>
        </div>
    </td>
    <td class="txt-center" data-column="Ngày đăng tải"><div>21:04 04/08/2026</div></td>
    <td class="txt-center" data-column="Đóng thầu"><div>10:00 13/08/2026</div></td>
</tr>
</table>
"""


def test_parse_dauthau_asia_rows_extracts_expected_fields():
    notices = parse_dauthau_asia_rows(_SAMPLE_HTML)
    assert len(notices) == 1
    n = notices[0]
    assert isinstance(n, TenderNotice)
    assert n.tbmt_id == "IB2600417491-00"
    assert n.package_name == "Chi phí xây dựng"
    assert "Nhân Cơ" in n.investor
    assert n.publish_date.isoformat() == "2026-08-04"
    assert n.close_date.isoformat() == "2026-08-13"
    assert n.source_url == "https://dauthau.asia/thongbao/moithau/chi-phi-xay-dung-1785812.html"
    assert n.package_value is None  # khong bi dat, khong co trong danh sach


def test_parse_dauthau_asia_rows_empty_html_returns_empty_list():
    assert parse_dauthau_asia_rows("<html><body>no table here</body></html>") == []


_DETAIL_HTML = """
<div class="bidding-detail-item">
    <div>
        <div class="c-tit">Chi tiết nguồn vốn</div>
        <div class="c-val">Nguồn vốn Ngân sách phường (nguồn sự nghiệp giáo dục)</div>
    </div>
    <div>
        <div class="c-tit">Hình thức LCNT</div>
        <div class="c-val">Đấu thầu rộng rãi</div>
    </div>
    <div>
        <div class="c-tit">Loại hợp đồng</div>
        <div class="c-val">Theo đơn giá cố định</div>
    </div>
    <div>
        <div class="c-tit">Lĩnh vực MSC <i class="fa fa-info-circle"></i></div>
        <div class="c-val">Xây lắp</div>
    </div>
    <div>
        <div class="c-tit">Giá gói thầu</div>
        <div class="c-val">Để xem đầy đủ thông tin mời bạn <a href="#">Đăng nhập</a> hoặc <a href="#">Đăng ký</a></div>
    </div>
</div>
"""


def _base_notice() -> TenderNotice:
    return TenderNotice(
        tbmt_id="IB1", package_name="Test", investor="Inv", source_url="https://dauthau.asia/x"
    )


def test_enrich_dauthau_asia_detail_fills_public_fields():
    enriched = enrich_dauthau_asia_detail(_base_notice(), _DETAIL_HTML)
    assert enriched.funding_source == "Ngân sách phường (nguồn sự nghiệp giáo dục)"
    assert enriched.selection_method == "Đấu thầu rộng rãi"
    assert enriched.contract_type == "Theo đơn giá cố định"
    assert enriched.package_type == "xây lắp"


def test_enrich_dauthau_asia_detail_does_not_fabricate_login_locked_fields():
    enriched = enrich_dauthau_asia_detail(_base_notice(), _DETAIL_HTML)
    assert enriched.package_value is None  # bị khoá sau đăng nhập, không được suy đoán


_KHLCNT_SAMPLE_HTML = """
<table class="bidding-table">
<tr>
    <td class="order-header" data-column="Tên dự án">
        <div>
            <a title="Bảo trì hệ thống phần mềm" href="/kehoach/luachon-nhathau/bao-tri-he-thong-phan-mem-2455731.html"><span class="plan-code">PL2600250241-00</span> Bảo trì hệ thống phần mềm</a>
        </div>
    </td>
    <td data-column="Chủ đầu tư">
        <div>
            <a title="Trường Cao đẳng Đà Lạt" href="/project-owner/truong-cao-dang-da-lat-122774/">  <span class="solicitor-code">vn5800371838</span>  Trường Cao đẳng Đà Lạt
            </a>
        </div>
    </td>
    <td class="txt-center" data-column="Ngày đăng tải"><div>15:55 08/08/2026</div></td>
    <td class="txt-center" data-column="Số gói thầu"><div>1</div></td>
</tr>
</table>
"""


def test_parse_dauthau_asia_khlcnt_rows_extracts_expected_fields():
    notices = parse_dauthau_asia_khlcnt_rows(_KHLCNT_SAMPLE_HTML)
    assert len(notices) == 1
    n = notices[0]
    assert isinstance(n, TenderNotice)
    assert n.tbmt_id == "PL2600250241-00"
    assert n.package_name == "Bảo trì hệ thống phần mềm"
    assert "Đà Lạt" in n.investor
    assert n.publish_date.isoformat() == "2026-08-08"
    assert n.close_date is None  # KHLCNT không có cột "đóng thầu", không được suy đoán
    assert n.source_url == "https://dauthau.asia/kehoach/luachon-nhathau/bao-tri-he-thong-phan-mem-2455731.html"


def test_parse_dauthau_asia_khlcnt_rows_empty_html_returns_empty_list():
    assert parse_dauthau_asia_khlcnt_rows("<html><body>no table here</body></html>") == []


def test_parse_dauthau_asia_khlcnt_rows_ignores_tbmt_rows():
    """Bảng có cùng class `bidding-table` với trang Thông báo mời thầu — phải phân biệt
    bằng `.plan-code` (KHLCNT) chứ không lẫn `.bidding-code` (TBMT)."""
    assert parse_dauthau_asia_khlcnt_rows(_SAMPLE_HTML) == []


def _khlcnt_detail_html(package_2_extra: str = "") -> str:
    # Mô phỏng đúng cấu trúc thật: nhãn cấp dự án dùng `.c-tit`/`.c-val`, nhãn cấp gói
    # thầu (trong khối đánh số `.cc-tit`) dùng `.c-tl`/`.c-vl` — xem docstring hàm enrich.
    return f"""
    <div>
        <div class="bidding-detail-item">
            <div><div class="c-tit">Chủ đầu tư</div><div class="c-val">Trường Cao đẳng Đà Lạt</div></div>
            <div><div class="c-tit">Bên mời thầu</div><div class="c-val">Trường Cao đẳng Đà Lạt</div></div>
        </div>
    </div>
    <div class="bidding-detail khlcnt-detail">
        <div class="bidding-detail-item"><div class="cc-tit"><span>1.</span>Gói 1</div></div>
        <div class="bidding-detail-item">
            <div class="c-tl">Chi tiết nguồn vốn</div>
            <div class="c-vl">Nguồn thu hợp pháp khác của đơn vị</div>
        </div>
        <div class="bidding-detail-item col-four">
            <div><div class="c-tl">Hình thức LCNT</div><div class="c-vl">Chỉ định thầu rút gọn</div></div>
            <div><div class="c-tl">Loại hợp đồng</div><div class="c-vl">Trọn gói</div></div>
        </div>
        <div class="bidding-detail-item col-four">
            <div><div class="c-tl">Thời gian thực hiện gói thầu</div><div class="c-vl">12 tháng</div></div>
            <div><div class="c-tl">Lĩnh vực</div><div class="c-vl">Phi tư vấn</div></div>
        </div>
        <div class="bidding-detail-item">
            <div class="c-tl">Giá gói thầu</div>
            <div class="c-vl">Để xem đầy đủ thông tin mời bạn <a href="#">Đăng nhập</a></div>
        </div>
        <div class="bidding-detail-item">
            <div class="c-tl">Tóm tắt công việc chính của gói thầu</div>
            <div class="c-vl">Bảo trì, hỗ trợ kỹ thuật, cập nhật phần mềm.</div>
        </div>
        {package_2_extra}
    </div>
    """


def test_enrich_dauthau_asia_khlcnt_detail_fills_project_and_first_package_fields():
    enriched = enrich_dauthau_asia_khlcnt_detail(_base_notice(), _khlcnt_detail_html())
    assert enriched.procuring_entity == "Trường Cao đẳng Đà Lạt"
    assert enriched.funding_source == "Nguồn thu hợp pháp khác của đơn vị"
    assert enriched.selection_method == "Chỉ định thầu rút gọn"
    assert enriched.contract_type == "Trọn gói"
    assert enriched.execution_time == "12 tháng"
    assert enriched.package_type == "phi tư vấn"


def test_enrich_dauthau_asia_khlcnt_detail_does_not_fabricate_login_locked_value():
    enriched = enrich_dauthau_asia_khlcnt_detail(_base_notice(), _khlcnt_detail_html())
    assert enriched.package_value is None


def test_enrich_dauthau_asia_khlcnt_detail_ignores_second_package_fields():
    """1 KHLCNT có thể liệt kê nhiều gói thầu — chỉ lấy dữ liệu của gói ĐẦU TIÊN, không
    trộn lẫn với gói thứ 2 trở đi (TenderNotice hiện chỉ mô hình hoá 1 gói/tài liệu)."""
    second_package = """
    <div class="bidding-detail-item"><div class="cc-tit"><span>2.</span>Gói 2</div></div>
    <div class="bidding-detail-item">
        <div class="c-tl">Loại hợp đồng</div>
        <div class="c-vl">Theo đơn giá cố định</div>
    </div>
    """
    enriched = enrich_dauthau_asia_khlcnt_detail(_base_notice(), _khlcnt_detail_html(second_package))
    assert enriched.contract_type == "Trọn gói"  # của gói 1, không phải "Theo đơn giá cố định" của gói 2


def test_extract_khlcnt_task_summary_returns_first_package_summary():
    summary = extract_khlcnt_task_summary(_khlcnt_detail_html())
    assert summary == "Bảo trì, hỗ trợ kỹ thuật, cập nhật phần mềm."


def test_extract_khlcnt_task_summary_returns_none_when_missing():
    assert extract_khlcnt_task_summary("<html><body>no summary here</body></html>") is None
