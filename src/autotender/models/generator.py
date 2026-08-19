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
    # Chương I — Chỉ dẫn nhà thầu (CDNT): Điều 26 Khoản 2 điểm a.
    "chuong_I.muc_1": {
        "chapter": "Chương I — Chỉ dẫn nhà thầu",
        "title": "Quy định chung",
        "query": "phạm vi áp dụng nguồn vốn tư cách hợp lệ của nhà thầu",
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
        "query": "thương thảo hợp đồng hoàn thiện ký kết hợp đồng điều kiện ký kết",
    },
    # Chương II — Bảng dữ liệu đấu thầu (BDL): số liệu cụ thể tương ứng Chương I — Điều 26
    # Khoản 2 điểm b. Bản chất là bảng tham số, không phải văn xuôi — xem `hint`.
    "chuong_II.muc_1": {
        "chapter": "Chương II — Bảng dữ liệu đấu thầu",
        "title": "Thông tin chung về gói thầu",
        "query": "tên gói thầu nguồn vốn hình thức lựa chọn nhà thầu phương thức đấu thầu",
        "hint": (
            "Trình bày dạng danh sách các mục dữ liệu — mỗi dòng là 1 mục tương ứng nội dung "
            "đã nêu ở Chương I (\"Tên mục dữ liệu: giá trị áp dụng cho gói thầu này\"), lấy "
            "giá trị từ trường thông tin gói thầu đã cho, KHÔNG viết văn xuôi diễn giải."
        ),
    },
    "chuong_II.muc_2": {
        "chapter": "Chương II — Bảng dữ liệu đấu thầu",
        "title": "Bảo đảm dự thầu, bảo đảm thực hiện hợp đồng và tiến độ đấu thầu",
        "query": "giá trị bảo đảm dự thầu bảo đảm thực hiện hợp đồng thời hạn hiệu lực",
        "hint": (
            "Trình bày dạng danh sách các mục dữ liệu (giá trị bảo đảm dự thầu, bảo đảm thực "
            "hiện hợp đồng, thời hạn hiệu lực hồ sơ dự thầu...), KHÔNG viết văn xuôi diễn giải."
        ),
    },
    "chuong_III.muc_1": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về năng lực và kinh nghiệm",
        "query": "tiêu chuẩn năng lực kinh nghiệm doanh thu nhà thầu",
    },
    "chuong_III.muc_2": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về kỹ thuật",
        "query": "tiêu chuẩn đánh giá kỹ thuật giải pháp nhân sự",
    },
    "chuong_III.muc_3": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Tiêu chuẩn đánh giá về giá",
        "query": "giá dự thầu xếp hạng đánh giá giá",
    },
    "chuong_III.muc_4": {
        "chapter": "Chương III — Tiêu chuẩn đánh giá E-HSDT",
        "title": "Yêu cầu về nhãn hiệu, xuất xứ hàng hóa",
        "query": "nhãn hiệu xuất xứ hàng hóa tương đương",
    },
    # Chương IV — Biểu mẫu mời thầu và dự thầu: mẫu để NHÀ THẦU điền khi nộp E-HSDT, KHÔNG
    # phải nội dung do bên mời thầu tự thuật — Điều 26 Khoản 2 điểm d. Xem `hint`.
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
    "chuong_V.muc_1": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Phạm vi cung cấp",
        "query": "phạm vi cung cấp lắp đặt đào tạo bảo hành",
    },
    "chuong_V.muc_2": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về thông số kỹ thuật",
        "query": "thông số kỹ thuật tương đương sản phẩm cụ thể",
    },
    "chuong_V.muc_3": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về bảo hành, bảo trì",
        "query": "bảo hành bảo trì phụ tùng thay thế",
    },
    "chuong_V.muc_4": {
        "chapter": "Chương V — Yêu cầu về kỹ thuật",
        "title": "Yêu cầu về tiến độ thực hiện",
        "query": "tiến độ thực hiện hợp đồng mốc bàn giao",
    },
    # Chương VI — Điều kiện chung của hợp đồng (ĐKC): quy định chung, không phụ thuộc đặc
    # thù từng gói thầu — Điều 26 Khoản 2 điểm e.
    "chuong_VI.muc_1": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Định nghĩa, phạm vi và loại hợp đồng",
        "query": "loại hợp đồng hồ sơ hợp đồng định nghĩa giải thích từ ngữ",
    },
    "chuong_VI.muc_2": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Quyền và nghĩa vụ của các bên",
        "query": "quyền nghĩa vụ chủ đầu tư nhà thầu nguyên tắc thực hiện hợp đồng",
    },
    "chuong_VI.muc_3": {
        "chapter": "Chương VI — Điều kiện chung của hợp đồng",
        "title": "Sửa đổi, thanh lý hợp đồng và xử lý vi phạm",
        "query": "sửa đổi hợp đồng thanh lý hợp đồng xử lý vi phạm",
    },
    # Chương VII — Điều kiện cụ thể của hợp đồng (ĐKCT): TUỲ BIẾN theo từng gói thầu (cùng
    # tinh thần Chương III/V — cần phán đoán, không phải boilerplate thuần như Chương VI).
    "chuong_VII.muc_1": {
        "chapter": "Chương VII — Điều kiện cụ thể của hợp đồng",
        "title": "Điều khoản cụ thể về giá, thanh toán và tiến độ",
        "query": "đồng tiền thanh toán tạm ứng thanh toán hợp đồng điều chỉnh giá",
    },
    "chuong_VII.muc_2": {
        "chapter": "Chương VII — Điều kiện cụ thể của hợp đồng",
        "title": "Điều khoản cụ thể về bảo hành, bảo trì và nghiệm thu",
        "query": "bảo hành bảo trì nghiệm thu chất lượng hàng hóa dịch vụ",
    },
    # Chương VIII — Biểu mẫu hợp đồng: mẫu để ký với NHÀ THẦU TRÚNG THẦU, KHÔNG phải nội
    # dung do bên mời thầu tự thuật — Điều 26 Khoản 2 điểm e. Xem `hint`.
    "chuong_VIII.muc_1": {
        "chapter": "Chương VIII — Biểu mẫu hợp đồng",
        "title": "Mẫu hợp đồng và phụ lục hợp đồng",
        "query": "hợp đồng đối với nhà thầu được lựa chọn nội dung hợp đồng ký kết",
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


# Các con số thông dụng trong quy định kỹ thuật / thời hạn / năm ban hành luật — không coi là số liệu tài chính sai lệch
_COMMON_SPEC_NUMBERS = {
    "10", "12", "15", "20", "24", "30", "45", "60", "90", "100", "120", "180", "360", "365",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2030",
    "32", "64", "128", "256", "512", "1024",
}


def verify_numeric_consistency(generated_text: str, fields: list[ExtractedField]) -> list[ComplianceFlag]:
    """So khớp số liệu tài chính & thông số quan trọng trong văn bản sinh ra với KHLCNT (Mục 2.2).

    Chỉ gắn cờ R4 với các số liệu tài chính / số liệu lớn (>100.000) hoặc số liệu đặc thù
    xuất hiện trong văn bản mà KHÔNG khớp với KHLCNT, tránh báo động giả với các thông số
    kỹ thuật phổ biến (bảo hành 24 tháng, RAM 64GB, năm 2025...).
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
    "Bạn là trợ lý soạn thảo hồ sơ mời thầu (HSMT) cho gói thầu phần mềm/CNTT tại Việt Nam. "
    "Hãy soạn nội dung mục được yêu cầu, dựa CHÍNH XÁC vào các trường thông tin gói thầu và "
    "trích đoạn văn bản pháp luật được cung cấp — KHÔNG bịa số liệu, KHÔNG tự suy diễn điều "
    "khoản pháp luật ngoài trích đoạn đã cho. Số liệu (giá gói thầu, thời gian, nguồn vốn...) "
    "PHẢI lấy nguyên văn từ trường thông tin gói thầu, không tự tính toán hay làm tròn khác đi. "
    "Ghi rõ nguồn căn cứ pháp lý (vd \"(Điều 44, Luật Đấu thầu 22/2023/QH15)\") khi áp dụng một "
    "quy định cụ thể. TUYỆT ĐỐI KHÔNG đưa ra tiêu chí mang tính chất hạn chế cạnh tranh hoặc "
    "tạo lợi thế cho một nhà thầu cụ thể — cấm nêu tên nhãn hiệu/xuất xứ cụ thể, cấm yêu cầu "
    "năng lực/doanh thu vượt ngưỡng hợp lý so với quy mô gói thầu, cấm mô tả thông số kỹ thuật "
    "chỉ một sản phẩm/nhà cung cấp mới đáp ứng được (vi phạm nguyên tắc cạnh tranh, công bằng, "
    "minh bạch — Điều 44 Luật Đấu thầu 22/2023/QH15). Đây là yêu cầu bắt buộc ngay khi soạn, "
    "không chỉ để hệ thống rà soát lại sau. Trả lời bằng tiếng Việt, văn phong hành chính, "
    "không thêm lời dẫn/kết thừa."
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

        model = self._cfg.get("claude_model") or self._cfg.get("model") or "claude-3-5-sonnet-20241022"
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
