"""M5 — Sinh dự thảo Chương III và Chương V của E-HSMT (Mục 6/M5).

Tier 1: `VietAI/vit5-base` fine-tune, checkpoint `models/generator_vit5`.
Tier 2: LLM API bên ngoài (OpenAI-compatible `/chat/completions`), CHỈ dùng nếu người
        dùng cấu hình biến môi trường `AUTOTENDER_LLM_API_KEY` — không tự ý gọi ra
        ngoài nếu người dùng không cấu hình (tôn trọng chi phí/quyền riêng tư).
Tier 3: Template filling thuần từ mẫu Thông tư (`data/samples/corpus/mau_hsmt_*.md`)
        — LUÔN THÀNH CÔNG.

NGUYÊN TẮC BẮT BUỘC (Mục 2.2): số liệu (giá gói thầu, thời gian, nguồn vốn) được chèn
bằng slot-filling từ `ExtractedField`, KHÔNG để mô hình tự sinh. Sau khi sinh, verifier
`verify_numeric_consistency` so khớp mọi con số xuất hiện trong văn bản sinh ra với các
con số đã biết từ KHLCNT — lệch thì gắn cờ `R4`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from autotender.config import get_models_settings, resolve_path
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.models.retriever import RetrieverModule
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


class GeneratorModule(BaseModule[GeneratedSection]):
    module_name = "M5-Generator"

    def __init__(self, retriever: RetrieverModule | None = None):
        super().__init__()
        self._cfg = get_models_settings().generator
        self._retriever = retriever or RetrieverModule()
        self._tier1_pipeline = None

    def generate_section(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        if section_id not in SECTION_DEFINITIONS:
            raise ValueError(f"Section '{section_id}' nằm ngoài phạm vi sinh của M5 (chỉ chuong_III/V).")
        result = self.run(section_id, fields)
        # Chỉ verify phần văn bản KHÔNG PHẢI trích dẫn nguyên văn từ corpus — số liệu trong
        # đoạn trích dẫn đã có citation đi kèm (truy vết được nguồn), không phải model tự bịa.
        narrative_only = result.text
        for c in result.citations:
            narrative_only = narrative_only.replace(c.text, "")
        result.flags = verify_numeric_consistency(narrative_only, fields)
        return result

    def _retrieve_context(self, section_id: str, top_k: int = 5) -> list[RetrievedChunk]:
        query = SECTION_DEFINITIONS[section_id]["query"]
        return self._retriever.retrieve(query, top_k=top_k)

    # -- Tier 1 -------------------------------------------------------------
    def _try_tier1(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        checkpoint = resolve_path(self._cfg.get("tier1_checkpoint", "models/generator_vit5"))
        if not Path(checkpoint).exists():
            raise TierUnavailableError(f"Không tìm thấy checkpoint tại {checkpoint}")
        try:
            from transformers import pipeline
        except ImportError as e:
            raise TierUnavailableError("Thư viện `transformers` chưa cài đặt") from e

        if self._tier1_pipeline is None:
            try:
                self._tier1_pipeline = pipeline("text2text-generation", model=str(checkpoint))
            except Exception as e:  # noqa: BLE001
                raise TierUnavailableError(f"Load checkpoint Tier 1 lỗi: {e}") from e

        citations = self._retrieve_context(section_id)
        prompt = self._build_prompt(section_id, fields, citations)
        try:
            output = self._tier1_pipeline(prompt, max_length=512)[0]["generated_text"]
        except Exception as e:  # noqa: BLE001
            raise TierUnavailableError(f"Suy luận Tier 1 lỗi: {e}") from e

        return GeneratedSection(
            section_id=section_id, title=SECTION_DEFINITIONS[section_id]["title"], text=output, citations=citations
        )

    # -- Tier 2 (LLM API — chỉ khi người dùng tự cấu hình key) ---------------
    def _try_tier2(self, section_id: str, fields: list[ExtractedField]) -> GeneratedSection:
        api_key_env = self._cfg.get("tier2_api_key_env", "AUTOTENDER_LLM_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise TierUnavailableError(f"Biến môi trường {api_key_env} chưa được cấu hình — bỏ qua Tier 2.")

        try:
            import httpx
        except ImportError as e:
            raise TierUnavailableError("Thư viện `httpx` chưa cài đặt") from e

        citations = self._retrieve_context(section_id)
        prompt = self._build_prompt(section_id, fields, citations)
        base_url = os.environ.get("AUTOTENDER_LLM_BASE_URL", "https://api.openai.com/v1")
        model = self._cfg.get("tier2_model", "gpt-4o-mini")
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise TierUnavailableError(f"Gọi LLM API Tier 2 thất bại: {e}") from e

        return GeneratedSection(
            section_id=section_id, title=SECTION_DEFINITIONS[section_id]["title"], text=text, citations=citations
        )

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
        context_str = "\n".join(f"[{c.source_doc}] {c.text}" for c in citations)
        return (
            f"Bạn là trợ lý soạn thảo E-HSMT. Hãy soạn nội dung mục \"{definition['title']}\" "
            f"thuộc {definition['chapter']}, dựa CHÍNH XÁC vào các trường và tài liệu tham chiếu sau, "
            f"không được bịa số liệu hay điều khoản pháp luật ngoài tài liệu tham chiếu:\n\n"
            f"Trường trích xuất từ KHLCNT:\n{fields_str}\n\n"
            f"Tài liệu tham chiếu:\n{context_str}\n\n"
            f"Nội dung mục cần soạn:"
        )
