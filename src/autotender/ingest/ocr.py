"""OCR cho các trang PDF scan (không có text layer) — Mục 6/M1.

Tier chính: PaddleOCR (`lang='vi'`). Fallback: VietOCR.
Nếu cả hai đều không cài được (rủi ro đã biết trên Windows, xem Mục 14 SPEC),
trả về thông báo rõ ràng thay vì crash toàn bộ pipeline ingest.

Xác nhận thực tế (2026-08-09): PaddleOCR 3.x đã đổi API constructor không tương thích ngược
(`use_angle_cls`/`show_log` bị loại bỏ) — CỐ Ý CHƯA sửa `_get_paddle_engine()` theo API mới,
vì đã kiểm chứng dù sửa đúng tham số, bước text-detection vẫn crash thật trên môi trường
Windows/CPU với lỗi engine oneDNN/PIR (`NotImplementedError:
ConvertPirAttribute2RuntimeAttribute not support ...`) — sự cố tương thích backend
PaddlePaddle 3.3 trên máy này, không phải lỗi gọi sai tham số. Sửa constructor chỉ dời điểm
lỗi từ "khởi tạo" (rẻ, phát hiện ngay, không tải gì) sang "mỗi lần OCR thật" (tốn ~200MB tải
model PaddleX lần đầu rồi vẫn crash) — tệ hơn hiện trạng. VietOCR là engine THẬT SỰ đang phục
vụ OCR (đã xác minh chạy đúng trên máy này); PaddleOCR luôn rơi xuống VietOCR ngay ở bước
khởi tạo, không ảnh hưởng người dùng cuối (chỉ 1 dòng log WARNING).
"""

from __future__ import annotations

from autotender.utils.logging import get_logger

logger = get_logger(__name__)

_paddle_engine = None
_paddle_unavailable = False
_vietocr_predictor = None
_vietocr_unavailable = False


def _get_paddle_engine():
    global _paddle_engine, _paddle_unavailable
    if _paddle_unavailable:
        return None
    if _paddle_engine is not None:
        return _paddle_engine
    try:
        from paddleocr import PaddleOCR

        _paddle_engine = PaddleOCR(lang="vi", use_angle_cls=True, show_log=False)
        return _paddle_engine
    except Exception as e:  # noqa: BLE001 — thư viện OCR có thể thiếu deps hệ thống (Windows)
        logger.warning("PaddleOCR không khởi tạo được: %s. Sẽ thử VietOCR.", e)
        _paddle_unavailable = True
        return None


def _get_vietocr_predictor():
    global _vietocr_predictor, _vietocr_unavailable
    if _vietocr_unavailable:
        return None
    if _vietocr_predictor is not None:
        return _vietocr_predictor
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config = Cfg.load_config_from_name("vgg_transformer")
        config["device"] = "cpu"
        _vietocr_predictor = Predictor(config)
        return _vietocr_predictor
    except Exception as e:  # noqa: BLE001
        logger.warning("VietOCR không khởi tạo được: %s.", e)
        _vietocr_unavailable = True
        return None


def ocr_image_bytes(image_bytes: bytes) -> str:
    """OCR một ảnh (PNG bytes) sang text tiếng Việt. Không bao giờ raise ra ngoài.

    Trả về `[CẦN NGƯỜI DÙNG BỔ SUNG: ...]` nếu không có engine OCR nào khả dụng —
    tuân thủ nguyên tắc "không bịa đặt" (Mục 2.2) khi hệ thống thật sự không đọc được nội dung.
    """
    engine = _get_paddle_engine()
    if engine is not None:
        try:
            import numpy as np
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = engine.ocr(np.array(image), cls=True)
            lines = [line[1][0] for block in result for line in block] if result else []
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            logger.warning("PaddleOCR lỗi khi xử lý ảnh: %s. Thử VietOCR.", e)

    predictor = _get_vietocr_predictor()
    if predictor is not None:
        try:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return predictor.predict(image)
        except Exception as e:  # noqa: BLE001
            logger.warning("VietOCR lỗi khi xử lý ảnh: %s.", e)

    return "[CẦN NGƯỜI DÙNG BỔ SUNG: không có engine OCR khả dụng (PaddleOCR/VietOCR chưa cài được)]"
