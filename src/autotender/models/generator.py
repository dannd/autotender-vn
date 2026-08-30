"""M5 — Sinh dự thảo trọn bộ E-HSMT (8 chương I-VIII, Mức 2, đề cương RAG+LLM).

Tier 1: Claude API (LLM có sẵn, KHÔNG tự train — đúng hướng redesign) + ngữ cảnh RAG
        (hybrid retrieval, `rag/hybrid_retriever.py`) — đường CHÍNH.
Tier 2: chưa dùng (chỗ dự phòng cho LLM khác nếu cần đổi nhà cung cấp sau này).
Tier 3: Template filling thuần (chèn trực tiếp trích dẫn Điều/Khoản liên quan nhất) —
        không cần API key, LUÔN THÀNH CÔNG.

NGUYÊN TẮC BẮT BUỘC (Mục 2.2): số liệu (giá gói thầu, thời gian, nguồn vốn) được chèn
bằng slot-filling từ `ExtractedField`, KHÔNG để mô hình tự sinh. Sau khi sinh, verifier
`verify_numeric_consistency` so khớp mọi con số xuất hiện trong văn bản sinh ra với các
con số đã biết từ KHLCNT — lệch thì gắn cờ `R4`.

`_retriever` chấp nhận bất kỳ đối tượng nào có `.retrieve(query, top_k=...)` (duck-typing,
không ép kiểu `HybridLegalRetriever`) — để test (`tests/test_generator.py`) truyền vào
retriever giả lập (trả về rỗng/kết quả cố định) mà không cần dựng `HybridLegalRetriever`
thật (chậm, cần corpus + embedding model).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from autotender.config import get_models_settings
from autotender.generation.llm_client import (
    ClaudeUnavailableError,
    LLMUnavailableError,
    call_claude,
    call_llm,
    is_configured as is_llm_configured,
)
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.rag.hybrid_retriever import HybridLegalRetriever
from autotender.schemas import ComplianceFlag, ExtractedField, RetrievedChunk
from autotender.utils.vn_text import format_vn_number

# Thứ tự các chương trong dict này LÀ thứ tự hiển thị/xuất bản chính thức (I→VIII) — xem
# `CHAPTER_TITLES` bên dưới (suy ra từ dict này, giữ thứ tự xuất hiện đầu tiên của mỗi
# chương). Ánh xạ 8 chương với Điều 26 Khoản 2 Nghị định 214/2025/NĐ-CP (a-g, xem
# models/compliance.py::REQUIRED_HSMT_COMPONENTS) — điểm g "hồ sơ, bản vẽ khác (nếu có)"
# không map 1-1 với 1 chương chuẩn nào nên không có mục riêng.
SECTION_DEFINITIONS: dict[str, dict[str, str]] = {
    # Chương I — Chỉ dẫn nhà thầu (CDNT): Điều 26 Khoản 2 điểm a Nghị định 214/2025/NĐ-CP & Nghị định 73/2019/NĐ-CP.
    "chuong_I.muc_1": {
        "chapter": "Chương I — Chỉ dẫn nhà thầu",
        "title": "Quy định chung",
        "query": "phạm vi áp dụng nguồn vốn tư cách hợp lệ của nhà thầu đầu tư ứng dụng công nghệ thông tin",
    },
    "chuong_I.muc_2": {
        "chapter": "Chương I — Chỉ dẫn nhà thầu",
        "title": "Chuẩn bị hồ sơ dự thầu",
        "query": "chuẩn bị hồ sơ dự thầu đơn dự thầu bảo đảm dự thầu hiệu lực hồ sơ dự thầu",
    },
    "chuong_I.muc_3": {
        "chapter": "Chương I — Chỉ dẫn nhà thầu",
        "title": "Nộp, mở và đánh giá hồ sơ dự thầu",
        "query": "nộp mở đánh giá hồ sơ dự thầu tính hợp lệ hồ sơ dự thầu",
    },
    "chuong_I.muc_4": {
        "chapter": "Chương I — Chỉ dẫn nhà thầu",
        "title": "Thương thảo, hoàn thiện và ký kết hợp đồng",
        "query": "thương thảo hợp đồng hoàn thiện ký kết hợp đồng điều kiện ký kết chuyển giao công nghệ",
    },
    # Chương II — Bảng dữ liệu đấu thầu (BDL): số liệu cụ thể tương ứng Chương I — Điều 26 Khoản 2 điểm b.
    "chuong_II.muc_1": {
        "chapter": "Chương II — Bảng dữ liệu đấu thầu",
        "title": "Thông tin chung về gói thầu",
        "query": "tên gói thầu nguồn vốn hình thức lựa chọn nhà thầu phương thức đấu thầu công nghệ thông tin",
        "hint": (
            "Trình bày dạng danh sách các mục dữ liệu — mỗi dòng là 1 mục tương ứng nội dung "
            "đã nêu ở Chương I (\"Tên mục dữ liệu: giá trị áp dụng cho gói thầu này\"), lấy "
            "giá trị từ trường thông tin gói thầu đã cho, KHÔNG viết văn xuôi diễn giải."
        ),
    },
    "chuong_II.muc_2": {
        "chapter": "Chương II — Bảng dữ liệu đấu thầu",
        "title": "Bảo đảm dự thầu, bảo đảm thực hiện hợp đồng và tiến độ đấu thầu",
        "query": "giá trị bảo đảm dự thầu bảo đảm thực hiện hợp đồng thời hạn hiệu lực tiến độ",
        "hint": (
            "Trình bày dạng danh sách các mục dữ liệu (giá trị bảo đảm dự thầu, bảo đảm thực "
            "hiện hợp đồng, thời hạn hiệu lực hồ sơ dự thầu...), KHÔNG viết văn xuôi diễn giải."
        ),
    },
    # Chương III — Tiêu chuẩn đánh giá E-HSDT
    "chuong_III.muc_1": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về năng lực và kinh nghiệm",
        "query": "tiêu chuẩn năng lực kinh nghiệm doanh thu nhà thầu hợp đồng tương tự phần mềm công nghệ thông tin",
        "hint": (
            "Xác định rõ yêu cầu về năng lực tài chính (doanh thu bình quân, tài sản ròng) và kinh nghiệm "
            "thực hiện hợp đồng tương tự về xây dựng, phát triển phần mềm/hệ thống CNTT theo đúng Thông tư 22/2024/TT-BKHĐT."
        ),
    },
    "chuong_III.muc_2": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về kỹ thuật và nhân sự chủ chốt",
        "query": "tiêu chuẩn đánh giá kỹ thuật nhân sự chủ chốt giám đốc dự án PMP kiến trúc sư giải pháp an toàn thông tin CISSP kiểm thử phần mềm",
        "hint": (
            "Quy định chi tiết tiêu chuẩn đánh giá giải pháp kỹ thuật, phương án triển khai, phương án bảo đảm an toàn thông tin "
            "theo Nghị định 85/2016/NĐ-CP và tiêu chuẩn nhân sự chủ chốt ngành CNTT: Giám đốc quản lý dự án (PMP/Scrum Master), "
            "Kiến trúc sư giải pháp (Solution Architect), Chuyên gia An toàn thông tin (CISSP/CEH/Security+), Kỹ sư phát triển chính."
        ),
    },
    "chuong_III.muc_3": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về giá",
        "query": "giá dự thầu phương pháp kết hợp kỹ thuật và giá xếp hạng đánh giá chi phí phần mềm",
    },
    "chuong_III.muc_4": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Yêu cầu về nhãn hiệu, xuất xứ hàng hóa",
        "query": "nhãn hiệu xuất xứ hàng hóa tương đương chuẩn mở công nghệ không khóa nhà cung cấp",
    },
    # Chương IV — Biểu mẫu mời thầu và dự thầu
    "chuong_IV.muc_1": {
        "chapter": "Chương IV — Biểu mẫu mời thầu và dự thầu",
        "title": "Mẫu đơn dự thầu và giấy uỷ quyền",
        "query": "đơn dự thầu đại diện hợp pháp ký tên đóng dấu giấy uỷ quyền",
        "hint": (
            "Soạn dưới dạng MẪU BIỂU để nhà thầu điền khi nộp hồ sơ dự thầu (không phải nội "
            "dung do bên mời thầu tự thuật) — dùng chỗ trống dạng \"[TÊN NHÀ THẦU]\", "
            "\"[NGÀY KÝ]\" cho thông tin chỉ nhà thầu mới biết, KHÔNG tự bịa tên nhà thầu cụ thể."
        ),
    },
    "chuong_IV.muc_2": {
        "chapter": "Chương IV — Biểu mẫu mời thầu và dự thầu",
        "title": "Mẫu bảo lãnh dự thầu và cam kết của nhà thầu",
        "query": "bảo đảm dự thầu thư bảo lãnh giấy chứng nhận bảo hiểm bảo lãnh",
        "hint": (
            "Soạn dưới dạng MẪU BIỂU để nhà thầu/tổ chức tín dụng điền — dùng chỗ trống dạng "
            "\"[TÊN NHÀ THẦU]\", \"[TÊN NGÂN HÀNG]\" cho thông tin chưa biết trước, KHÔNG tự bịa."
        ),
    },
    # Chương V — Yêu cầu về kỹ thuật
    "chuong_V.muc_1": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Phạm vi cung cấp và phát triển phần mềm",
        "query": "phạm vi cung cấp thiết kế chi tiết phần mềm nội bộ khảo sát lập trình kiểm thử UAT đào tạo chuyển giao công nghệ Nghị định 73 2019",
        "hint": (
            "Mô tả chi tiết phạm vi các hạng mục công việc phát triển phần mềm theo Nghị định 73/2019/NĐ-CP và Nghị định 82/2024/NĐ-CP: "
            "khảo sát, thiết kế chi tiết, lập trình, kiểm thử đơn vị, kiểm thử tích hợp, kiểm thử chấp nhận UAT, đào tạo và chuyển giao mã nguồn."
        ),
    },
    "chuong_V.muc_2": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về thông số kỹ thuật và An toàn thông tin",
        "query": "thông số kỹ thuật kiến trúc phần mềm API RESTful an toàn thông tin theo cấp độ Nghị định 85 2016 TCVN 11930 mã hóa TLS",
        "hint": (
            "Quy định đầy đủ yêu cầu chức năng, yêu cầu phi chức năng (hiệu năng, tải đồng thời, độ sẵn sàng) và yêu cầu bắt buộc về "
            "bảo đảm an toàn hệ thống thông tin theo cấp độ theo Nghị định số 85/2016/NĐ-CP, tiêu chuẩn TCVN 11930:2017 (xác thực đa yếu tố MFA, mã hóa TLS 1.3/AES-256, kiểm thử lỗ hổng bảo mật OWASP)."
        ),
    },
    "chuong_V.muc_3": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về bảo hành, bảo trì và cam kết mức độ dịch vụ (SLA)",
        "query": "bảo hành bảo trì phần mềm cam kết mức độ dịch vụ SLA thời gian khắc phục sự cố hỗ trợ kỹ thuật",
        "hint": (
            "Quy định thời gian bảo hành tối thiểu 12-24 tháng theo Nghị định 73/2019/NĐ-CP, cam kết mức độ dịch vụ hỗ trợ (SLA) "
            "phản hồi trong 02-04 giờ đối với sự cố nghiêm trọng, hỗ trợ kỹ thuật 24/7 và nâng cấp bản vá bảo mật định kỳ."
        ),
    },
    "chuong_V.muc_4": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về tiến độ thực hiện và kế hoạch bàn giao",
        "query": "tiến độ thực hiện hợp đồng mốc bàn giao sản phẩm phần mềm nghiệm thu",
    },
    # Chương VI — Điều kiện chung của hợp đồng (ĐKC)
    "chuong_VI.muc_1": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Định nghĩa, phạm vi và loại hợp đồng",
        "query": "loại hợp đồng hồ sơ hợp đồng định nghĩa giải thích từ ngữ dịch vụ phần mềm công nghệ thông tin",
    },
    "chuong_VI.muc_2": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Quyền và nghĩa vụ của các bên về Sở hữu trí tuệ và Mã nguồn",
        "query": "quyền nghĩa vụ chủ đầu tư nhà thầu sở hữu trí tuệ bản quyền toàn bộ mã nguồn source code cơ sở dữ liệu Nghị định 73 2019",
        "hint": (
            "Quy định rõ: Chủ đầu tư là chủ sở hữu duy nhất đối với toàn bộ mã nguồn (source code), cơ sở dữ liệu và tài liệu kỹ thuật "
            "hình thành từ hợp đồng theo Điều 55 Nghị định 73/2019/NĐ-CP; nhà thầu không được giữ bản quyền độc quyền hay cài đặt mã độc, backdoor."
        ),
    },
    "chuong_VI.muc_3": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Sửa đổi, thanh lý hợp đồng và xử lý vi phạm",
        "query": "sửa đổi hợp đồng thanh lý hợp đồng xử lý vi phạm phạt vi phạm hợp đồng",
    },
    # Chương VII — Điều kiện cụ thể của hợp đồng (ĐKCT)
    "chuong_VII.muc_1": {
        "chapter": "Chương VII — Điều kiện cụ thể của hợp đồng",
        "title": "Điều khoản cụ thể về giá, thanh toán và tiến độ",
        "query": "đồng tiền thanh toán tạm ứng thanh toán hợp đồng điều chỉnh giá mốc nghiệm thu",
    },
    "chuong_VII.muc_2": {
        "chapter": "Chương VII — Điều kiện cụ thể của hợp đồng",
        "title": "Điều khoản cụ thể về bảo hành, bảo mật và bàn giao toàn bộ mã nguồn",
        "query": "bảo hành bảo trì nghiệm thu bàn giao mã nguồn tài liệu thiết kế an toàn thông tin bảo mật dữ liệu",
        "hint": (
            "Quy định chi tiết điều kiện nghiệm thu bàn giao trọn bộ: mã nguồn sạch có chú thích, kịch bản CI/CD, cơ sở dữ liệu, "
            "tài liệu kiến trúc và hướng dẫn quản trị; cam kết bảo mật thông tin (NDA) và nghĩa vụ bảo hành 12 tháng."
        ),
    },
    # Chương VIII — Biểu mẫu hợp đồng
    "chuong_VIII.muc_1": {
        "chapter": "Chương VIII — Biểu mẫu hợp đồng",
        "title": "Mẫu hợp đồng và phụ lục hợp đồng",
        "query": "hợp đồng đối với nhà thầu được lựa chọn nội dung hợp đồng ký kết chuyển giao phần mềm",
        "hint": (
            "Soạn dưới dạng MẪU HỢP ĐỒNG (điều khoản khung) để điền khi ký với nhà thầu trúng "
            "thầu — dùng chỗ trống dạng \"[TÊN NHÀ THẦU TRÚNG THẦU]\", \"[NGÀY KÝ HỢP ĐỒNG]\" "
            "cho thông tin chỉ có sau khi có kết quả lựa chọn nhà thầu, KHÔNG tự bịa."
        ),
    },
}

# Tiêu đề hiển thị mỗi chương, suy ra từ SECTION_DEFINITIONS (1 nguồn duy nhất — trước đây
# `export/docx.py` và `export/pdf.py` mỗi nơi tự giữ 1 bản `_CHAPTER_TITLES` riêng, dễ lệch
# nhau khi thêm chương mới). Thứ tự dict = thứ tự xuất hiện đầu tiên của mỗi chương trong
# SECTION_DEFINITIONS ở trên = thứ tự hiển thị/xuất bản chính thức (I→VIII).
CHAPTER_TITLES: dict[str, str] = {
    sid.split(".")[0]: meta["chapter"] for sid, meta in SECTION_DEFINITIONS.items()
}

_PLACEHOLDER = "[CẦN NGƯỜI DÙNG BỔ SUNG: {desc}]"

_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")

# Số thứ tự mục kiểu "1.1", "2.3" (Claude hay dùng khi soạn danh sách có cấu trúc, xác
# nhận thực tế khi chạy live) — tiếng Việt dùng dấu phẩy cho số thập phân, không phải dấu
# chấm, nên "N.N" (1 chữ số . 1 chữ số) gần như luôn là số thứ tự mục, không phải số liệu
# nghiệp vụ thật. Loại khỏi kiểm tra để tránh cờ R4 giả.
_OUTLINE_MARKER_RE = re.compile(r"^\d\.\d$")

# Số trong TRÍCH DẪN CĂN CỨ PHÁP LÝ nội tuyến (vd "Điều 26 Nghị định 214/2025/NĐ-CP") —
# khác bản Tier 3 cũ (chèn nguyên văn `c.text`, lọc bằng cách xoá đúng chuỗi đó), Claude
# (Tier 1) diễn giải/tóm tắt nên không copy verbatim `c.text` — obviously KHÔNG khớp bằng
# string replace. Số Điều/Khoản/năm ban hành trong các cụm trích dẫn này KHÔNG PHẢI số
# liệu gói thầu (giá/thời gian/nguồn vốn) nên không cần đối chiếu KHLCNT — nguồn đã hiển
# thị tường minh ở panel trích dẫn riêng. Phát hiện thực tế khi chạy live với Claude API.
_CITATION_REF_RE = re.compile(
    r"Điều\s+\d+|[Kk]hoản\s+\d+|"
    r"Nghị định\s+(?:số\s+)?\d+/\d{4}(?:/NĐ-CP)?|"
    r"Luật(?:\s+Đấu\s+thầu)?\s+(?:số\s+)?\d+/\d{4}/QH\d+|"
    r"Thông tư\s+(?:số\s+)?\d+/\d{4}(?:/TT-[A-ZĐ]+)?"
)


def _strip_citation_references(text: str) -> str:
    return _CITATION_REF_RE.sub("", text)


@dataclass
class GeneratedSection:
    section_id: str
    title: str
    text: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    flags: list[ComplianceFlag] = field(default_factory=list)


def _field_value(fields: list[ExtractedField], name: str) -> str | None:
    for f in fields:
        if f.name == name:
            return f.value
    return None


def _format_currency(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return value
    return format_vn_number(int(digits)) + " đồng"


# Các con số thông dụng trong quy định kỹ thuật / thời hạn / năm ban hành luật / thông số CNTT phổ biến — không coi là số liệu tài chính sai lệch
_COMMON_SPEC_NUMBERS = {
    "1", "2", "3", "4", "5", "7", "10", "12", "14", "15", "20", "24", "30", "45", "60", "90", "100", "120", "180", "360", "365",
    "2016", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2030",
    "32", "64", "128", "256", "512", "1024", "2048", "4096",
    "80", "443", "8080", "3306", "5432", "27017",
    "73", "82", "85", "214", "22", "30", "11930", "27001",
    "99", "999", "9999",  # SLA uptime 99.9%, 99.99%
}


def verify_numeric_consistency(generated_text: str, fields: list[ExtractedField]) -> list[ComplianceFlag]:
    """So khớp số liệu tài chính & thông số quan trọng trong văn bản sinh ra với KHLCNT (Mục 2.2).

    Chỉ gắn cờ R4 với các số liệu tài chính / số liệu lớn (>100.000) hoặc số liệu đặc thù
    xuất hiện trong văn bản mà KHÔNG khớp với KHLCNT, tránh báo động giả với các thông số
    kỹ thuật phổ biến (bảo hành 24 tháng, RAM 64GB, năm 2025, cổng 443, SLA 99.9%...).
    """
    known_numbers: set[str] = set()
    for f in fields:
        for m in _NUMBER_RE.finditer(f.value):
            normalized = re.sub(r"[.,]", "", m.group())
            if normalized:
                known_numbers.add(normalized)

    flags: list[ComplianceFlag] = []
    for m in _NUMBER_RE.finditer(generated_text):
        raw = m.group()
        if _OUTLINE_MARKER_RE.match(raw):  # "1.1", "2.3"... số thứ tự mục, không phải số liệu
            continue
        normalized = re.sub(r"[.,]", "", raw)
        if len(normalized) < 2:  # bỏ qua số đơn lẻ (số thứ tự mục, v.v.)
            continue
        if normalized in _COMMON_SPEC_NUMBERS and normalized not in known_numbers:
            continue
        if normalized not in known_numbers:
            flags.append(
                ComplianceFlag(
                    rule_code="R4",
                    severity="cao",
                    sentence=raw,
                    explanation=(
                        f"Số liệu '{raw}' xuất hiện trong văn bản sinh ra nhưng không khớp với "
                        "bất kỳ giá trị nào đã trích xuất từ KHLCNT — cần người dùng kiểm tra lại."
                    ),
                    evidence=[],
                    confidence=0.7,
                )
            )
    return flags


_SYSTEM_PROMPT = (
    "Bạn là trợ lý soạn thảo hồ sơ mời thầu (HSMT) chuyên nghiệp cho các gói thầu phần mềm và công nghệ thông tin (CNTT) tại Việt Nam. "
    "Hãy soạn nội dung mục được yêu cầu, dựa CHÍNH XÁC vào các trường thông tin gói thầu và trích đoạn văn bản pháp luật được cung cấp "
    "(Luật Đấu thầu số 22/2023/QH15, Nghị định 214/2025/NĐ-CP, Nghị định 73/2019/NĐ-CP, Nghị định 82/2024/NĐ-CP và Nghị định 85/2016/NĐ-CP). "
    "KHÔNG bịa số liệu, KHÔNG tự suy diễn điều khoản pháp luật ngoài trích đoạn đã cho. Số liệu (giá gói thầu, thời gian thực hiện, nguồn vốn...) "
    "PHẢI lấy nguyên văn từ trường thông tin gói thầu, không tự tính toán hay làm tròn khác đi. "
    "Ghi rõ nguồn căn cứ pháp lý cụ thể khi áp dụng một quy định (ví dụ: \"(Điều 44 Luật Đấu thầu 22/2023/QH15)\", \"(Điều 55 Nghị định 73/2019/NĐ-CP)\", \"(Nghị định 85/2016/NĐ-CP)\"). "
    "Đối với gói thầu phần mềm/CNTT, bắt buộc thể hiện rõ các yêu cầu về an toàn thông tin theo cấp độ, tiêu chuẩn nhân sự CNTT chuẩn mực (PMP, Scrum Master, CISSP, Solution Architect), "
    "và điều khoản bắt buộc bàn giao 100% mã nguồn sạch (clean source code), cơ sở dữ liệu cho Chủ đầu tư. "
    "TUYỆT ĐỐI KHÔNG đưa ra tiêu chí mang tính chất hạn chế cạnh tranh hoặc tạo lợi thế cho một nhà thầu cụ thể — cấm nêu tên nhãn hiệu/xuất xứ cụ thể mà không có cụm \"hoặc tương đương\", "
    "cấm yêu cầu năng lực/doanh thu vượt ngưỡng hợp lý so với quy mô gói thầu, cấm mô tả thông số kỹ thuật chỉ một sản phẩm/nhà cung cấp độc quyền mới đáp ứng được. "
    "Trả lời bằng tiếng Việt, văn phong hành chính công vụ trang trọng, cấu trúc Markdown rõ ràng, không thêm lời dẫn/kết thừa."
)


class GeneratorModule(BaseModule[GeneratedSection]):
    module_name = "M5-Generator"

    def __init__(self, retriever: HybridLegalRetriever | None = None):
        super().__init__()
        self._cfg = get_models_settings().generator
        self._retriever = retriever or HybridLegalRetriever()

    def generate_section(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        if section_id not in SECTION_DEFINITIONS:
            raise ValueError(f"Section '{section_id}' không tồn tại trong SECTION_DEFINITIONS.")
        result = self.run(section_id, fields)
        # Chỉ verify phần văn bản KHÔNG PHẢI trích dẫn — số liệu trong đoạn trích dẫn đã có
        # citation đi kèm (truy vết được nguồn), không phải model tự bịa.
        narrative_only = result.text
        for c in result.citations:
            narrative_only = narrative_only.replace(c.text, "")
        narrative_only = _strip_citation_references(narrative_only)
        result.flags = verify_numeric_consistency(narrative_only, fields)
        return result

    def _retrieve_context(self, section_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        query = SECTION_DEFINITIONS[section_id]["query"]
        return self._retriever.retrieve(query, top_k=top_k)

    def _retrieve_context_reranked(self, section_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        query = SECTION_DEFINITIONS[section_id]["query"]
        return self._retriever.retrieve_reranked(query, top_k=top_k)

    # -- Tier 1: LLM Gateway / Claude API + RAG (đường chính) ------------------------------
    def _try_tier1(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        if not is_llm_configured():
            raise TierUnavailableError("LLM API Key chưa được cấu hình — bỏ qua truy xuất+rerank tốn thời gian.")

        citations = self._retrieve_context_reranked(section_id)
        if not citations:
            raise TierUnavailableError("Không truy xuất được trích đoạn nào liên quan cho mục này.")

        from autotender.config import get_app_settings
        app_cfg = get_app_settings()
        model = (
            os.environ.get("LLM_MODEL")
            or getattr(app_cfg.llm_gateway, "default_model", None)
            or self._cfg.get("claude_model")
            or self._cfg.get("model")
            or "claude-sonnet-4-5-20250929"
        )
        prompt = self._build_prompt(section_id, fields, citations, self._retriever)
        try:
            text = call_claude(
                system=_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=model,
                max_tokens=self._cfg.get("max_tokens", 4096),
            )
        except (LLMUnavailableError, ClaudeUnavailableError) as e:
            raise TierUnavailableError(str(e)) from e

        return GeneratedSection(
            section_id=section_id, title=SECTION_DEFINITIONS[section_id]["title"], text=text, citations=citations
        )

    # -- Tier 2: dự phòng (chưa dùng) -----------------------------------------
    def _try_tier2(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        raise TierUnavailableError("Tier 2 chưa được cấu hình cho module sinh.")

    # -- Tier 3 (bắt buộc luôn thành công) -----------------------------------
    def _try_tier3(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        definition = SECTION_DEFINITIONS[section_id]
        citations = self._retrieve_context(section_id)

        package_name = _field_value(fields, "PACKAGE_NAME") or _PLACEHOLDER.format(desc="tên gói thầu")
        investor = _field_value(fields, "INVESTOR") or _PLACEHOLDER.format(desc="chủ đầu tư")
        value = _format_currency(_field_value(fields, "VALUE")) or _PLACEHOLDER.format(desc="giá gói thầu")
        duration = _field_value(fields, "DURATION") or _PLACEHOLDER.format(desc="thời gian thực hiện")
        funding = _field_value(fields, "FUNDING") or _PLACEHOLDER.format(desc="nguồn vốn")

        intro = (
            f"Căn cứ gói thầu \"{package_name}\" do {investor} làm chủ đầu tư, "
            f"giá gói thầu {value}, nguồn vốn {funding}, thời gian thực hiện {duration}, "
            f"nội dung mục \"{definition['title']}\" quy định như sau:"
        )

        body_parts = [c.text for c in citations[:2]]
        body = "\n\n".join(body_parts) if body_parts else _PLACEHOLDER.format(desc="nội dung tham chiếu từ corpus")

        text = f"{intro}\n\n{body}"
        return GeneratedSection(section_id=section_id, title=definition["title"], text=text, citations=citations)

    @staticmethod
    def _build_prompt(
        section_id: str,
        fields: list[ExtractedField],
        citations: list[RetrievedChunk],
        retriever: HybridLegalRetriever,
    ) -> str:
        definition = SECTION_DEFINITIONS[section_id]
        fields_str = "\n".join(f"- {f.name}: {f.value}" for f in fields)
        # Gửi TRỌN Điều (không chỉ đoạn Khoản đã khớp truy hồi) làm ngữ cảnh cho Claude —
        # xem `HybridLegalRetriever.expand_to_parent_article`. Chỉ áp ở đây (ngữ cảnh cho
        # LLM), KHÔNG áp cho retrieval/rerank/citation hiển thị UI hay Tier 3 template.
        context_str = "\n\n".join(
            f"[{c.source_doc}]\n{retriever.expand_to_parent_article(c)}" for c in citations
        )
        hint = definition.get("hint")
        hint_str = f"\n\nGợi ý trình bày riêng cho mục này: {hint}" if hint else ""
        return (
            f"Hãy soạn nội dung mục \"{definition['title']}\" thuộc {definition['chapter']}.\n\n"
            f"Trường thông tin gói thầu (đã trích xuất từ KHLCNT):\n{fields_str}\n\n"
            f"Trích đoạn văn bản pháp luật liên quan:\n\n{context_str}"
            f"{hint_str}\n\n"
            f"Nội dung mục cần soạn (chỉ trả về nội dung, không thêm tiêu đề mục lặp lại):"
        )
