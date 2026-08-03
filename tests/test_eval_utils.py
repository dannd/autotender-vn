import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_utils import fields_to_bio_tags  # noqa: E402
from autotender.schemas import ExtractedField  # noqa: E402


def test_fields_to_bio_tags_marks_correct_span():
    text = "Tên gói thầu: Mua sắm máy chủ. Chủ đầu tư: Sở Y tế."
    fields = [
        ExtractedField(name="PACKAGE_NAME", value="Mua sắm máy chủ", confidence=0.9, source="regex",
                        char_start=text.index("Mua sắm máy chủ"), char_end=text.index("Mua sắm máy chủ") + len("Mua sắm máy chủ")),
    ]
    tags = fields_to_bio_tags(text, fields)
    assert "B-PACKAGE_NAME" in tags
    # Token cuối "chủ." dính dấu chấm nên vượt char_end của span 1 ký tự -> không được gắn nhãn
    # (giới hạn đã biết của tokenizer tách theo khoảng trắng, giống scripts/build_dataset.py).
    assert tags.count("I-PACKAGE_NAME") == 2


def test_fields_to_bio_tags_ignores_fields_without_span():
    text = "Chủ đầu tư: Sở Y tế."
    fields = [ExtractedField(name="INVESTOR", value="Sở Y tế", confidence=0.9, source="manual")]
    tags = fields_to_bio_tags(text, fields)
    assert all(t == "O" for t in tags)
