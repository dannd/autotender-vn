from autotender.utils.vn_text import merge_broken_lines, normalize_nfc, split_sentences


def test_normalize_nfc_combines_decomposed_diacritics():
    decomposed = "Nguyên Thị Hương"  # NFD-ish sequence
    result = normalize_nfc(decomposed)
    assert result == unicodedata_normalize(decomposed)


def unicodedata_normalize(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", s)


def test_merge_broken_lines_joins_sentence_without_terminator():
    text = "Gói thầu này có giá trị\nrất lớn.\nChủ đầu tư là Sở Y tế."
    merged = merge_broken_lines(text)
    assert "giá trị rất lớn." in merged


def test_split_sentences_does_not_split_after_abbreviation():
    text = "Theo TT. 79/2025/TT-BTC, gói thầu phải tuân thủ. Đây là câu thứ hai."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Theo TT. 79/2025")
