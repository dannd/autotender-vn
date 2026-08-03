from pathlib import Path

import pytest

from autotender.crawler.sources import LocalSampleSource, TenderSourceError


def test_local_sample_source_reads_bundled_samples():
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    source = LocalSampleSource(samples_dir)
    notices = list(source.fetch("2025-01-01", "2026-12-31", max_records=20))
    assert len(notices) == 20
    assert all(n.tbmt_id for n in notices)


def test_local_sample_source_respects_max_records():
    samples_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    source = LocalSampleSource(samples_dir)
    notices = list(source.fetch("2025-01-01", "2026-12-31", max_records=3))
    assert len(notices) == 3


def test_local_sample_source_missing_file_raises(tmp_path):
    source = LocalSampleSource(tmp_path)
    with pytest.raises(TenderSourceError):
        list(source.fetch("2025-01-01", "2026-12-31", max_records=5))
