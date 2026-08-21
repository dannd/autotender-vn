"""Mức 1 — Hỏi-đáp có trích dẫn (đề cương RAG+LLM), nền tảng bắt buộc trước Mức 2/3.

Khác cơ chế 3-tier "tự train dần" của bản cũ (Tier 1 checkpoint fine-tune → Tier 2
pretrained zero-shot → Tier 3 rule-based): ở đây dùng LLM có sẵn (Claude API) làm đường
CHÍNH ngay từ đầu — không cần huấn luyện. Vẫn giữ khung `BaseModule` 3 tầng để nhất quán
với phần còn lại của hệ thống và đảm bảo nguyên tắc "luôn chạy được":

Tier 1: Claude API + ngữ cảnh RAG (hybrid retrieval) — CHÍNH.
Tier 2: chưa dùng (chỗ dự phòng cho LLM khác nếu cần đổi nhà cung cấp sau này).
Tier 3: liệt kê trích dẫn không qua LLM — không cần API key, LUÔN THÀNH CÔNG.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from autotender.config import get_models_settings
from autotender.generation.law_classifier import DEFAULT_CLASSIFIER_MODEL, classify_relevant_law_ids
from autotender.generation.llm_client import (
    ClaudeUnavailableError,
    LLMUnavailableError,
    call_claude,
    call_llm,
    is_configured as is_llm_configured,
)
from autotender.generation.query_rewrite import DEFAULT_REWRITE_MODEL, rewrite_query
from autotender.models.base import BaseModule, TierUnavailableError
from autotender.rag.hybrid_retriever import HybridLegalRetriever
from autotender.schemas import QAAnswer, RetrievedChunk

_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý về đấu thầu tại Việt Nam, hỗ trợ soạn hồ sơ mời thầu (HSMT) cho "
    "gói thầu phần mềm/CNTT. Chỉ trả lời DỰA VÀO các trích đoạn văn bản pháp luật được cung "
    "cấp dưới đây — KHÔNG dùng kiến thức ngoài, KHÔNG tự suy diễn số liệu hay điều khoản. "
    "Nếu các trích đoạn không đủ căn cứ để trả lời, PHẢI nói rõ là không tìm thấy căn cứ, "
    "không được đoán hay bịa. Mỗi ý trong câu trả lời phải ghi rõ nguồn (vd \"(Điều 44, Luật "
    "Đấu thầu 22/2023/QH15)\"). Trả lời bằng tiếng Việt, ngắn gọn, đúng thuật ngữ pháp lý."
)


def _build_user_prompt(question: str, citations: list[RetrievedChunk], retriever: HybridLegalRetriever) -> str:
    # Gửi TRỌN Điều (không chỉ đoạn Khoản đã khớp truy hồi) làm ngữ cảnh cho Claude — xem
    # `HybridLegalRetriever.expand_to_parent_article`. Chỉ áp cho ngữ cảnh LLM, không đổi
    # citation hiển thị UI (vẫn dùng `c.text`/`c.source_doc` gốc, xem `QAAnswer.citations`).
    context = "\n\n".join(f"[{c.source_doc}]\n{retriever.expand_to_parent_article(c)}" for c in citations)
    return f"Các trích đoạn văn bản pháp luật liên quan:\n\n{context}\n\nCâu hỏi: {question}"


class LegalQAModule(BaseModule[QAAnswer]):
    module_name = "QA-Mức1"

    def __init__(self, retriever: HybridLegalRetriever | None = None):
        super().__init__()
        self._cfg = get_models_settings().qa
        self._retriever = retriever or HybridLegalRetriever(
            model_key=self._cfg.get("embedding_model_key", "vi_bi_encoder")
        )

    def ask(self, question: str) -> QAAnswer:
        return self.run(question)

    def _retrieve_citations(
        self, question: str, *, allow_rerank: bool = True, law_ids: set[str] | None = None
    ) -> list[RetrievedChunk]:
        top_k = self._cfg.get("top_k_citations", 5)
        candidate_k = self._cfg.get("candidate_k", 50)
        if allow_rerank and self._cfg.get("use_rerank", True):
            return self._retriever.retrieve_reranked(question, top_k=top_k, candidate_k=candidate_k, law_ids=law_ids)
        return self._retriever.retrieve(question, top_k=top_k, candidate_k=candidate_k, law_ids=law_ids)

    # -- Tier 1: LLM Gateway / Claude API + RAG (đường chính) ------------------------------
    def _try_tier1(self, question: str) -> QAAnswer:
        if not is_llm_configured():
            raise TierUnavailableError("LLM API Key chưa cấu hình — bỏ qua truy xuất+rerank tốn thời gian.")

        # HyDE-lite: chuẩn hoá thuật ngữ câu hỏi TRƯỚC KHI truy hồi (không đổi câu hỏi hiển
        # thị/trả lời cho người dùng, chỉ đổi câu dùng để tìm kiếm) — xem
        # `generation/query_rewrite.py`. Chỉ áp ở Mức 1 (câu hỏi tự do từ người dùng), không
        # áp cho Mức 2 (`generator.py`, câu truy vấn đã viết sẵn bằng thuật ngữ chuẩn).
        retrieval_query = question
        # Mặc định False — đo thật (configs/models.yaml, docs/DATA_CARD.md Mục 12.2) cho
        # thấy rewrite làm GIẢM chất lượng truy hồi trên tập eval hiện tại.
        if self._cfg.get("use_query_rewrite", False):
            retrieval_query = rewrite_query(question, model=self._cfg.get("rewrite_model", DEFAULT_REWRITE_MODEL))

        # Metadata filtering theo loại văn bản — xem generation/law_classifier.py và
        # docs/DATA_CARD.md Mục 12.3 (đo cả trần lý thuyết lẫn bộ phân loại thật).
        law_ids = None
        if self._cfg.get("use_law_id_filter", False):
            law_ids = classify_relevant_law_ids(
                question, model=self._cfg.get("classifier_model", DEFAULT_CLASSIFIER_MODEL)
            )

        citations = self._retrieve_citations(retrieval_query, law_ids=law_ids)
        if not citations:
            raise TierUnavailableError("Không truy xuất được trích đoạn nào liên quan.")

        from autotender.config import get_app_settings
        app_cfg = get_app_settings()
        model = (
            os.environ.get("LLM_MODEL")
            or getattr(app_cfg.llm_gateway, "default_model", None)
            or self._cfg.get("claude_model")
            or self._cfg.get("model")
            or "claude-sonnet-4-5-20250929"
        )
        try:
            answer_text = call_claude(
                system=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(question, citations, self._retriever),
                model=model,
                max_tokens=self._cfg.get("max_tokens", 1024),
            )
        except (LLMUnavailableError, ClaudeUnavailableError) as e:
            raise TierUnavailableError(str(e)) from e

        return QAAnswer(
            question=question, answer=answer_text, citations=citations,
            model_used=model, generated_at=datetime.now(timezone.utc),
        )

    # -- Tier 2: dự phòng (chưa dùng) -----------------------------------------
    def _try_tier2(self, question: str) -> QAAnswer:
        raise TierUnavailableError("Tier 2 chưa được cấu hình cho module Hỏi-đáp.")

    # -- Tier 3: liệt kê trích dẫn không qua LLM, luôn thành công ------------
    def _try_tier3(self, question: str) -> QAAnswer:
        citations = self._retrieve_citations(question, allow_rerank=False)
        if citations:
            answer = (
                "Không thể gọi LLM API Gateway (thiếu cấu hình hoặc lỗi kết nối) — dưới đây là các "
                "trích đoạn văn bản pháp luật liên quan nhất, người dùng tự đối chiếu:\n\n"
                + "\n\n".join(f"({c.source_doc})\n{c.text}" for c in citations)
            )
        else:
            answer = "Không tìm thấy trích đoạn văn bản pháp luật nào liên quan đến câu hỏi này trong kho tri thức."
        return QAAnswer(
            question=question, answer=answer, citations=citations,
            model_used="template", generated_at=datetime.now(timezone.utc),
        )
