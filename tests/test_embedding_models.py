import numpy as np

from autotender.rag.embedding_models import encode_texts


class _FakeTokenizer:
    """1 'token' = 1 từ (tách theo khoảng trắng) — đủ để kiểm tra logic cắt cửa sổ mà
    không cần tokenizer subword thật."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, token_ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(token_ids)


class _FakeModel:
    """Giả lập SentenceTransformer — mỗi cửa sổ được encode thành vector [số từ, hash cố
    định] để kiểm tra: (a) văn bản ngắn không bị chia cửa sổ, (b) văn bản dài được chia và
    kết quả là trung bình đã chuẩn hoá của các cửa sổ, không phải embedding của bản bị cắt."""

    def __init__(self, max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, show_progress_bar=False, batch_size=32):
        return np.asarray([[len(t.split()), (hash(t) % 1000) / 1000.0] for t in texts], dtype="float64")


def test_short_text_uses_single_window_directly():
    model = _FakeModel(max_seq_length=10)  # budget = 8 token sau khi trừ [CLS]/[SEP]
    text = "một hai ba bốn"  # 4 từ, trong ngân sách
    result = encode_texts(model, [text])

    expected = model.encode([text])[0]
    norm = np.linalg.norm(expected)
    np.testing.assert_allclose(result[0], expected)  # không mean-pool/renormalize vì chỉ 1 cửa sổ
    assert norm > 0  # sanity: vector không rỗng


def test_long_text_is_split_and_mean_pooled_not_silently_truncated():
    model = _FakeModel(max_seq_length=6)  # budget = 4 token
    long_text = " ".join(f"từ{i}" for i in range(20))  # 20 từ, vượt xa ngân sách 4

    result = encode_texts(model, [long_text])

    # Nếu bị cắt âm thầm (chỉ lấy 4 từ đầu), kết quả sẽ khớp encode([4 từ đầu]) — PHẢI khác.
    truncated_only = model.encode([" ".join(long_text.split()[:4])])[0]
    assert not np.allclose(result[0], truncated_only)

    # Vector kết quả phải là unit-norm (đã chuẩn hoá lại sau mean-pool).
    assert np.isclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)


def test_batch_preserves_order_and_length_for_mixed_short_and_long():
    model = _FakeModel(max_seq_length=6)
    texts = ["ngắn", " ".join(f"từ{i}" for i in range(15)), "cũng ngắn"]

    result = encode_texts(model, texts)

    assert result.shape[0] == 3
    # 2 văn bản ngắn không bị pool -> khớp trực tiếp encode() của riêng nó.
    np.testing.assert_allclose(result[0], model.encode([texts[0]])[0])
    np.testing.assert_allclose(result[2], model.encode([texts[2]])[0])
