import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_dataset import build_classifier_dataset, build_ner_dataset  # noqa: E402
from autotender.schemas import TenderNotice  # noqa: E402


def _sample_notice() -> TenderNotice:
    return TenderNotice(
        tbmt_id="IB999",
        package_name="Mua sắm máy tính xách tay",
        investor="Sở Y tế tỉnh X",
        package_value=1_000_000_000,
        funding_source="Ngân sách nhà nước",
        selection_method="đấu thầu rộng rãi",
        contract_type="Trọn gói",
        package_type="hàng hóa",
        execution_time="30 ngày",
        source_url="https://example.com",
    )


def test_build_ner_dataset_labels_known_fields():
    records = build_ner_dataset([_sample_notice()])
    assert len(records) == 1
    tags = records[0]["tags"]
    assert any(t == "B-PACKAGE_NAME" for t in tags)
    assert any(t == "B-INVESTOR" for t in tags)
    assert any(t == "B-VALUE" for t in tags)
    assert len(tags) == len(records[0]["tokens"])


def test_build_classifier_dataset_maps_label():
    records = build_classifier_dataset([_sample_notice()])
    assert records[0]["label"] == "hang_hoa"
