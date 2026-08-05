"""M5 — Sinh dự thảo Chương III và Chương V của E-HSMT (Mức 2, đề cương RAG+LLM).

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
không ép kiểu `HybridLegalRetriever`) — để `scripts/evaluate.py` (ablation Phase 1 cũ) vẫn
truyền được `RetrieverModule` (BM25 Tier 3) làm ví dụ "no-RAG" (retrieve trả về rỗng) mà
không cần sửa lại.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from autotender.config import get_models_settings
from autotender.generation.claude_client import ClaudeUnavailableError, call_claude
from autotender.generation.claude_client import is_configured as is_claude_configured
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.rag.hybrid_retriever import HybridLegalRetriever
from autotender.schemas import ComplianceFlag, ExtractedField, RetrievedChunk

SECTION_DEFINITIONS: dict[str, dict[str, str]] = {
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
    return f"{int(digits):,}".replace(",", ".") + " đồng"


def verify_numeric_consistency(generated_text: str, fields: list[ExtractedField]) -> list[ComplianceFlag]:
    """So khớp mọi con số trong văn bản sinh ra với số liệu đã biết từ KHLCNT (Mục 2.2).

    Con số nào xuất hiện trong `generated_text` mà KHÔNG khớp bất kỳ giá trị số nào
    trong `fields` sẽ bị gắn cờ R4 (nghi ngờ số liệu bịa/sai lệch).
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
    "quy định cụ thể. Trả lời bằng tiếng Việt, văn phong hành chính, không thêm lời dẫn/kết thừa."
)


class GeneratorModule(BaseModule[GeneratedSection]):
    module_name = "M5-Generator"

    def __init__(self, retriever: HybridLegalRetriever | None = None):
        super().__init__()
        self._cfg = get_models_settings().generator
        self._retriever = retriever or HybridLegalRetriever()

    def generate_section(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        if section_id not in SECTION_DEFINITIONS:
            raise ValueError(f"Section '{section_id}' nằm ngoài phạm vi sinh của M5 (chỉ chuong_III/V).")
        result = self.run(section_id, fields)
        # Chỉ verify phần văn bản KHÔNG PHẢI trích dẫn — số liệu trong đoạn trích dẫn đã có
        # citation đi kèm (truy vết được nguồn), không phải model tự bịa. Bản Tier 3 chèn
        # nguyên văn `c.text` nên xoá bằng string replace; bản Tier 1 (Claude) diễn giải/tóm
        # tắt nên còn cần xoá thêm các cụm trích dẫn nội tuyến (`_strip_citation_references`).
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

    # -- Tier 1: Claude API + RAG (đường chính) ------------------------------
    def _try_tier1(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        if not is_claude_configured():
            raise TierUnavailableError("ANTHROPIC_API_KEY chưa cấu hình — bỏ qua truy xuất+rerank tốn thời gian.")

        citations = self._retrieve_context_reranked(section_id)
        if not citations:
            raise TierUnavailableError("Không truy xuất được trích đoạn nào liên quan cho mục này.")

        model = self._cfg.get("claude_model", "claude-sonnet-5")
        prompt = self._build_prompt(section_id, fields, citations)
        try:
            text = call_claude(
                system=_SYSTEM_PROMPT, user_prompt=prompt, model=model,
                max_tokens=self._cfg.get("max_tokens", 1536),
            )
        except ClaudeUnavailableError as e:
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
    def _build_prompt(section_id: str, fields: list[ExtractedField], citations: list[RetrievedChunk]) -> str:
        definition = SECTION_DEFINITIONS[section_id]
        fields_str = "\n".join(f"- {f.name}: {f.value}" for f in fields)
        context_str = "\n\n".join(f"[{c.source_doc}]\n{c.text}" for c in citations)
        return (
            f"Hãy soạn nội dung mục \"{definition['title']}\" thuộc {definition['chapter']}.\n\n"
            f"Trường thông tin gói thầu (đã trích xuất từ KHLCNT):\n{fields_str}\n\n"
            f"Trích đoạn văn bản pháp luật liên quan:\n\n{context_str}\n\n"
            f"Nội dung mục cần soạn (chỉ trả về nội dung, không thêm tiêu đề mục lặp lại):"
        )
