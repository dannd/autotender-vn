import json
import logging

from autotender.utils.logging import _JsonFormatter, get_logger


def _make_record(**kwargs) -> logging.LogRecord:
    defaults = dict(
        name="autotender.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="xin chào %s", args=("thế giới",), exc_info=None,
    )
    defaults.update(kwargs)
    return logging.LogRecord(**defaults)


def test_json_formatter_produces_valid_json_with_expected_fields():
    formatter = _JsonFormatter()
    record = _make_record()

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "autotender.test"
    assert payload["message"] == "xin chào thế giới"
    assert "ts" in payload


def test_json_formatter_includes_extra_fields():
    formatter = _JsonFormatter()
    record = _make_record()
    record.doc_id = "doc_123"
    record.username = "dan"

    payload = json.loads(formatter.format(record))

    assert payload["doc_id"] == "doc_123"
    assert payload["username"] == "dan"


def test_json_formatter_includes_exception_info():
    formatter = _JsonFormatter()
    try:
        raise ValueError("loi gia lap")
    except ValueError:
        import sys
        record = _make_record(exc_info=sys.exc_info())

    payload = json.loads(formatter.format(record))

    assert "loi gia lap" in payload["exc_info"]


def test_get_logger_returns_named_logger():
    logger = get_logger("autotender.test.module")
    assert logger.name == "autotender.test.module"
