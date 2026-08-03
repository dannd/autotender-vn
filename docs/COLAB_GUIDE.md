# Hướng dẫn chạy training trên Google Colab

Repo: https://github.com/dannd/autotender-vn

## 0. Về dataset

Dataset dùng để huấn luyện là **20 bản ghi tổng hợp** tại `data/samples/tender_notices.jsonl`
(đã commit trong repo) + corpus mẫu tại `data/samples/corpus/*.md`. Đây KHÔNG phải dữ liệu
crawl thật — xem lý do và giới hạn đầy đủ trong [`DATA_CARD.md`](DATA_CARD.md).

`data/processed/*.jsonl` (dataset NER/classifier đã qua distant-supervision) **không nằm
trong git** (bị `.gitignore`) vì đây là dữ liệu sinh ra từ script, không phải dữ liệu gốc —
mỗi notebook sẽ tự chạy `scripts/build_dataset.py` để tái tạo lại, không cần bạn tải gì thêm.

Với chỉ 20 mẫu, mô hình sẽ **overfit** — checkpoint sinh ra chủ yếu để (a) xác nhận toàn bộ
pipeline huấn luyện chạy đúng, (b) có Tier 1 thật để demo cơ chế fallback, KHÔNG phải để có
độ chính xác cao trên dữ liệu thật. Khi có dữ liệu crawl thật lớn hơn, lặp lại đúng quy trình
này với `data/samples/tender_notices.jsonl` được thay bằng dữ liệu thật.

## 1. Mở notebook trên Colab

Có 4 notebook, độc lập với nhau (mỗi cái chạy trên 1 runtime riêng của Colab):

| Notebook | Module | Thời gian ước tính (T4) |
|---|---|---|
| `notebooks/01_train_ner.ipynb` | M2 — NER | ~5-10 phút |
| `notebooks/02_train_classifier.ipynb` | M3 — Classifier | ~5-10 phút |
| `notebooks/03_train_retriever.ipynb` | M4 — Retriever | ~5-10 phút |
| `notebooks/04_train_generator.ipynb` | M5 — Generator | ~10-15 phút |

Cách mở nhanh nhất — dán URL sau vào trình duyệt (thay `01_train_ner` bằng tên notebook
tương ứng):

```
https://colab.research.google.com/github/dannd/autotender-vn/blob/main/notebooks/01_train_ner.ipynb
```

Hoặc: vào [colab.research.google.com](https://colab.research.google.com) → **File → Open
notebook → GitHub** → dán `dannd/autotender-vn` → chọn notebook cần chạy.

## 2. Bật GPU

Trong Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU** → Save.
(M5 Generator nên dùng T4 trở lên; M2/M3/M4 chạy được cả trên CPU nhưng chậm hơn nhiều.)

## 3. Chạy từng cell theo thứ tự (Runtime → Run all, hoặc Shift+Enter từng ô)

Mỗi notebook đã tự chứa đủ các bước:
1. `!pip install ...` — cài thư viện cần thiết.
2. `!git clone https://github.com/dannd/autotender-vn.git` — tải code + dataset mẫu.
3. (Chỉ 01, 02) `!python scripts/build_dataset.py` — sinh lại dataset NER/classifier.
4. Load dữ liệu, tokenize, huấn luyện, đánh giá (F1/macro-F1/ROUGE-L tuỳ notebook).
5. Lưu checkpoint vào `/content/models/<tên_module>/`.
6. **Cell cuối cùng: nén thành `.zip` và tự động tải về máy** (dùng
   `google.colab.files.download`, không cần mount Google Drive).

## 4. Đưa checkpoint vào repo local

Sau khi tải file `.zip` về máy (ví dụ `ner_phobert.zip`), giải nén đúng vào:

```
autotender-vn/models/ner_phobert/
autotender-vn/models/classifier_phobert/
autotender-vn/models/retriever_bi_encoder/
autotender-vn/models/generator_vit5/
```

(Tên thư mục phải khớp `configs/models.yaml` — đã đặt sẵn đúng tên ở trên.)

## 5. Xác nhận Tier 1 đã hoạt động

```bash
cd autotender-vn
pytest -q                      # vẫn phải pass (Tier 3 vẫn là fallback)
streamlit run app/main.py      # vào Trang 6 — badge tier phải chuyển 🟢 Tier 1
python scripts/evaluate.py     # ghi lại reports/metrics.json với số liệu Tier 1 thật
```

Trang 6 (Bảng điều khiển Model) sẽ hiển thị checkpoint đã tồn tại (✅) và tier đang chạy
chuyển từ 🔵 Tier 3 sang 🟢 Tier 1 sau khi module đó được gọi ít nhất 1 lần trong phiên.

## 6. Ablation (Mục 10 SPEC)

- Notebook 01 có sẵn mục "Ablation: PhoBERT vs XLM-R" ở cuối — đổi `MODEL_NAME` thành
  `'xlm-roberta-base'` và chạy lại các cell huấn luyện để so sánh.
- Notebook 03 có sẵn phần so sánh BM25 baseline vs bi-encoder fine-tuned (Recall@5, MRR@10).

## Lưu ý bảo mật

Không dán Personal Access Token hay bất kỳ khoá bí mật nào vào trong notebook — các
notebook này public trên GitHub, mọi thứ dán vào sẽ bị lộ. Việc `git clone` không cần
token vì repo là public.
