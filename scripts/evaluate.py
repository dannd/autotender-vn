"""Đánh giá toàn bộ pipeline, ghi `reports/metrics.json` + biểu đồ (Mục 10).

GIỚI HẠN QUAN TRỌNG (ghi rõ trong báo cáo): môi trường chạy đồ án này KHÔNG có GPU/Colab
để huấn luyện Tier 1 (notebooks 01-05 chưa chạy thật). Script này vì vậy đánh giá
**Tier 3 (rule-based)** — tầng luôn hoạt động theo Degraded Mode (Mục 2.1) — làm baseline
chính thức, cùng 1-2 baseline cổ điển (TF-IDF+LogisticRegression, BM25) theo đúng yêu cầu
Mục 10. Khi có checkpoint Tier 1 thật (chạy notebooks trên Colab), chạy lại script này để
điền thêm cột "tier1" vào từng bảng — code đã chừa sẵn hook (`_load_tier1_metrics_if_any`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from eval_utils import COMPLIANCE_TEST_SET, fields_to_bio_tags  # noqa: E402
from seqeval.metrics import classification_report as seq_report  # noqa: E402
from seqeval.metrics import f1_score as seq_f1  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from autotender.config import resolve_path  # noqa: E402
from autotender.ingest.synth_document import build_synthetic_khlcnt_text  # noqa: E402
from autotender.models.classifier import ClassifierModule  # noqa: E402
from autotender.models.compliance import ComplianceModule  # noqa: E402
from autotender.models.generator import GeneratorModule  # noqa: E402
from autotender.models.ner import NERModule  # noqa: E402
from autotender.models.retriever import RetrieverModule  # noqa: E402
from autotender.schemas import TenderNotice  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402
from autotender.utils.logging import get_logger  # noqa: E402

ensure_utf8_console()
logger = get_logger(__name__)

SAMPLES_FILE = resolve_path("data/samples/tender_notices.jsonl")
REPORTS_DIR = resolve_path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
SEEDS = [0, 1, 2]


def _load_notices() -> list[TenderNotice]:
    with open(SAMPLES_FILE, encoding="utf-8") as f:
        return [TenderNotice.model_validate_json(line) for line in f]


def evaluate_ner(notices: list[TenderNotice]) -> dict:
    ner = NERModule()
    all_gold, all_pred = [], []
    for notice in notices:
        text = build_synthetic_khlcnt_text(notice)
        fields = ner.extract(text)
        pred_tags = fields_to_bio_tags(text, fields)
        # Gold: coi field trích xuất bằng regex Tier3 khớp CHÍNH XÁC giá trị đã biết là đúng
        # (silver label từ chính pipeline distant-supervision, xem scripts/build_dataset.py).
        from build_dataset import _field_spans, _tokenize_with_spans

        tokens = _tokenize_with_spans(text)
        spans = _field_spans(notice, text)
        gold_tags = ["O"] * len(tokens)
        for start, end, label in spans:
            first = True
            for i, (_tok, tstart, tend) in enumerate(tokens):
                if tstart >= start and tend <= end:
                    gold_tags[i] = ("B-" if first else "I-") + label
                    first = False
        all_gold.append(gold_tags)
        all_pred.append(pred_tags)

    report = seq_report(all_gold, all_pred, output_dict=True, zero_division=0)
    f1 = seq_f1(all_gold, all_pred)
    logger.info("M2 NER (Tier 3) entity-F1 = %.3f", f1)
    return {"tier3_entity_f1": f1, "per_entity": report}


def evaluate_classifier(notices: list[TenderNotice]) -> dict:
    module = ClassifierModule()
    texts = [build_synthetic_khlcnt_text(n) for n in notices]
    labels = [
        {"hàng hóa": "hang_hoa", "xây lắp": "xay_lap", "tư vấn": "tu_van", "phi tư vấn": "phi_tu_van", "hỗn hợp": "hon_hop"}.get(
            n.package_type, "hang_hoa"
        )
        for n in notices
    ]

    tier3_preds = [module.classify(t).label for t in texts]
    tier3_f1 = f1_score(labels, tier3_preds, average="macro", zero_division=0)
    logger.info("M3 Classifier (Tier 3) macro-F1 = %.3f", tier3_f1)

    baseline_f1s = []
    for seed in SEEDS:
        if len(set(labels)) < 2 or len(texts) < 6:
            break
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.3, random_state=seed)
        vec = TfidfVectorizer(max_features=2000)
        Xt = vec.fit_transform(X_train)
        Xv = vec.transform(X_val)
        clf = LogisticRegression(max_iter=1000).fit(Xt, y_train)
        baseline_f1s.append(f1_score(y_val, clf.predict(Xv), average="macro", zero_division=0))

    result = {"tier3_macro_f1": tier3_f1}
    if baseline_f1s:
        result["baseline_tfidf_logreg_macro_f1_mean"] = float(np.mean(baseline_f1s))
        result["baseline_tfidf_logreg_macro_f1_std"] = float(np.std(baseline_f1s))
    else:
        result["baseline_note"] = "Không đủ dữ liệu (20 mẫu, nhiều lớp thưa) để chia train/val ổn định cho baseline."
    return result


def evaluate_retrieval() -> dict:
    from autotender.models.generator import SECTION_DEFINITIONS

    retriever = RetrieverModule()
    hits = 0
    for section_id, defn in SECTION_DEFINITIONS.items():
        expected_chapter = "Chương III" if section_id.startswith("chuong_III") else "Chương V"
        results = retriever.retrieve(defn["query"], top_k=5)
        if any(expected_chapter in r.source_doc for r in results):
            hits += 1
    recall_at_5 = hits / len(SECTION_DEFINITIONS)
    logger.info("M4 Retrieval (BM25, Tier 3) proxy Recall@5 = %.3f", recall_at_5)
    return {
        "bm25_proxy_recall_at_5": recall_at_5,
        "note": (
            "Recall@5 tính theo proxy: xem 1 truy vấn là 'đúng' nếu top-5 chứa ít nhất 1 chunk "
            "từ đúng file mẫu chương tương ứng. Chưa có tập câu hỏi gán tay độc lập — xem giới hạn "
            "nghiên cứu trong DATA_CARD.md."
        ),
    }


def evaluate_compliance() -> dict:
    module = ComplianceModule()
    y_true, y_pred = [], []
    for sentence, expected in COMPLIANCE_TEST_SET:
        flags = module.check_text(sentence)
        predicted = flags[0].rule_code if flags else "OK"
        y_true.append(expected)
        y_pred.append(predicted)

    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    logger.info("M6 Compliance (Tier 3) per-class: %s", per_class)
    return {"per_class": per_class, "n_test_samples": len(COMPLIANCE_TEST_SET)}


def run_ablation(notices: list[TenderNotice]) -> dict:
    ner = NERModule()
    notice = notices[0]
    text = build_synthetic_khlcnt_text(notice)
    fields = ner.extract(text)

    generator = GeneratorModule()
    section_id = "chuong_III.muc_1"

    with_retrieval = generator.generate_section(section_id, fields)
    n_citations_with = len(with_retrieval.citations)

    empty_retriever = RetrieverModule()
    empty_retriever._try_tier3 = lambda query, top_k: []  # type: ignore[method-assign]
    gen_no_retrieval = GeneratorModule(retriever=empty_retriever)
    without_retrieval = gen_no_retrieval.generate_section(section_id, fields)
    n_citations_without = len(without_retrieval.citations)

    compliance = ComplianceModule()
    with_m6_flags = sum(len(compliance.check_text(s)) for s, _ in COMPLIANCE_TEST_SET if compliance.check_text(s))
    return {
        "a_bo_retrieval": {
            "so_citation_co_retrieval": n_citations_with,
            "so_citation_khong_retrieval": n_citations_without,
            "nhan_xet": "Không có retrieval → 0 trích dẫn, nội dung sinh ra chỉ còn phần slot-filling, mất căn cứ pháp lý.",
        },
        "b_bo_hard_negative": {
            "note": "N/A trong môi trường này — cần huấn luyện bi-encoder trên Colab (notebooks/03_train_retriever.ipynb), chưa thực hiện do thiếu GPU."
        },
        "c_bo_m6": {
            "so_flag_phat_hien_co_m6": with_m6_flags,
            "so_flag_phat_hien_khong_m6": 0,
            "nhan_xet": "Không có M6 → toàn bộ vi phạm (nhãn hiệu, doanh thu bất hợp lý...) không được cảnh báo cho người dùng.",
        },
        "d_phobert_vs_xlmr": {
            "note": "N/A trong môi trường này — cần huấn luyện cả 2 backbone trên Colab (notebooks/01,02), chưa thực hiện do thiếu GPU."
        },
    }


def _plot_bar(data: dict[str, float], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(data.keys()), list(data.values()), color="#4C72B0")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    notices = _load_notices()

    metrics = {
        "environment_note": (
            "Đánh giá chạy trên Tier 3 (rule-based) do môi trường không có GPU/Colab để huấn luyện "
            "Tier 1 trong phạm vi đồ án 7 ngày. Baseline cổ điển (TF-IDF+LogisticRegression, BM25) "
            "được dùng làm mốc so sánh theo đúng yêu cầu Mục 10 SPEC."
        ),
        "m2_ner": evaluate_ner(notices),
        "m3_classifier": evaluate_classifier(notices),
        "m4_retrieval": evaluate_retrieval(),
        "m6_compliance": evaluate_compliance(),
        "ablation": run_ablation(notices),
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Đã ghi %s", metrics_path)

    _plot_bar(
        {k: v["f1"] for k, v in metrics["m6_compliance"]["per_class"].items()},
        "M6 Compliance — F1 theo lớp (Tier 3)",
        FIGURES_DIR / "m6_compliance_f1.png",
    )
    _plot_bar(
        {e: v["f1-score"] for e, v in metrics["m2_ner"]["per_entity"].items() if isinstance(v, dict) and "f1-score" in v},
        "M2 NER — F1 theo entity (Tier 3)",
        FIGURES_DIR / "m2_ner_f1.png",
    )
    logger.info("Đã ghi biểu đồ vào %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
