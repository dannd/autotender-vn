from autotender.crawler.parser import parse_dauthau_asia_rows
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
