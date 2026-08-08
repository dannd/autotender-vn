"""Entrypoint Streamlit — AutoTender-VN (Mục 7).

Định nghĩa điều hướng qua `st.navigation`/`st.Page` thay vì để Streamlit tự sinh menu
từ tên file trong `pages/` — cách tự động đó hiển thị tên file (không dấu, gạch dưới
thay khoảng trắng, ví dụ "Nap KHLCNT") thay vì tiếng Việt có dấu thật.
"""

from __future__ import annotations

import streamlit as st

home = st.Page("home.py", title="Trang chủ", icon="🏠", default=True)

soan_thao_pages = [
    st.Page("pages/1_Thu_thap_du_lieu.py", title="Thu thập dữ liệu", icon="📥"),
    st.Page("pages/2_Nap_KHLCNT.py", title="Nạp KHLCNT", icon="📝"),
    st.Page("pages/3_Soan_thao_HSMT.py", title="Soạn thảo HSMT", icon="🧾"),
    st.Page("pages/4_Kiem_tra_tuan_thu.py", title="Kiểm tra tuân thủ", icon="✅"),
    st.Page("pages/5_Xuat_va_In.py", title="Xuất và In", icon="📤"),
]
phan_tich_pages = [
    st.Page("pages/7_Hoi_dap.py", title="Hỏi-đáp (Mức 1)", icon="💬"),
    st.Page("pages/6_Bang_dieu_khien_Model.py", title="Bảng điều khiển", icon="📊"),
    st.Page("pages/8_Danh_gia.py", title="Đánh giá", icon="📈"),
]

pg = st.navigation(
    {
        "": [home],
        "Soạn thảo HSMT": soan_thao_pages,
        "Hỏi-đáp & Phân tích": phan_tich_pages,
    }
)
pg.run()
