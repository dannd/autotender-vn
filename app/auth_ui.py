"""Cổng đăng nhập cho toàn bộ ứng dụng — gọi từ `common.py::init_page`, NGAY SAU
`st.set_page_config` (phải là lệnh Streamlit đầu tiên của trang, xem giới hạn của
Streamlit) và TRƯỚC khi trang vẽ bất kỳ nội dung nào khác.

Trước khi có tầng này, bất kỳ ai vào được địa chỉ mạng của ứng dụng đều sửa/duyệt/xuất
tài liệu được, và `approved_by` bị gán cứng bằng 1 chuỗi tên cố định trong code
(`app/pages/3_Soan_thao_HSMT.py`) bất kể ai thật sự bấm duyệt — không thể quy trách nhiệm
đúng người cho một công cụ có giá trị pháp lý trong đấu thầu.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autotender.auth.store import AuthStore  # noqa: E402
from autotender.config import get_app_settings, resolve_path  # noqa: E402

_SESSION_KEY = "auth_user"
_TEST_USER = {"username": "test", "display_name": "Test User", "role": "admin"}


@st.cache_resource
def _get_auth_store() -> AuthStore:
    settings = get_app_settings()
    return AuthStore(resolve_path(settings.app.auth_db_path))


def current_user() -> dict | None:
    """`{"username", "display_name", "role"}` của người đang đăng nhập, hoặc `None` nếu
    chưa đăng nhập (chỉ xảy ra khi gọi TRƯỚC `require_login()`, vốn đã `st.stop()` nếu
    chưa đăng nhập — các trang bình thường không cần tự kiểm tra `None`)."""
    return st.session_state.get(_SESSION_KEY)


def _render_login_form() -> None:
    st.title("📑 AutoTender-VN")
    st.caption("Đăng nhập để tiếp tục — mọi thao tác soạn thảo/duyệt đều được gắn với tài khoản của bạn.")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập", type="primary")

    if submitted:
        user = _get_auth_store().verify_password(username.strip(), password)
        if user is None:
            st.error("Sai tên đăng nhập hoặc mật khẩu.")
        else:
            st.session_state[_SESSION_KEY] = user
            st.rerun()


def require_login() -> dict:
    """Chặn trang lại (`st.stop()`) cho tới khi đăng nhập thành công. Gọi ngay sau
    `st.set_page_config()` trong `init_page()`. Trả về thông tin người dùng khi đã đăng
    nhập (dùng cho trang nào cần đọc trực tiếp, thường dùng `current_user()` cho gọn)."""
    # Cho phép test (AppTest) bỏ qua màn đăng nhập — cùng tinh thần với override
    # AUTOTENDER_DB_PATH ở get_store(): cách ly test khỏi phụ thuộc bên ngoài (ở đây là
    # cần tài khoản thật), không ảnh hưởng hành vi khi chạy thật.
    if os.environ.get("AUTOTENDER_SKIP_AUTH") == "1":
        st.session_state[_SESSION_KEY] = _TEST_USER
        return _TEST_USER
    user = current_user()
    if user is not None:
        return user
    _render_login_form()
    st.stop()
    raise RuntimeError("unreachable")  # st.stop() không return — chỉ để type checker yên tâm


def render_user_badge_and_logout() -> None:
    """Hiển thị tên người dùng đang đăng nhập + nút đăng xuất trong sidebar — gọi sau
    `require_login()` nên luôn có user hợp lệ."""
    user = current_user()
    if user is None:
        return
    with st.sidebar:
        st.caption(f"👤 **{user['display_name']}** ({user['role']})")
        if st.button("Đăng xuất", key="logout_button", use_container_width=True):
            del st.session_state[_SESSION_KEY]
            st.rerun()
