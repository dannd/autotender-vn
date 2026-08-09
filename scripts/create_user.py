"""Tạo tài khoản đăng nhập cho AutoTender-VN (xem `app/auth_ui.py`, `src/autotender/auth/store.py`).

Ví dụ:
  python scripts/create_user.py --username dan --display-name "Nguyễn Đình Đán" --role admin
  python scripts/create_user.py --username vu --display-name "Vũ" --role editor
    (không truyền --password sẽ được hỏi qua terminal, không hiện ký tự gõ)
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autotender.auth.store import AuthStore, UserAlreadyExistsError  # noqa: E402
from autotender.config import get_app_settings, resolve_path  # noqa: E402
from autotender.utils.console import ensure_utf8_console  # noqa: E402

ensure_utf8_console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--password", default=None, help="Bỏ qua để được hỏi an toàn qua terminal")
    parser.add_argument("--role", default="editor", choices=["editor", "admin"])
    args = parser.parse_args()

    password = args.password or getpass.getpass("Mật khẩu: ")
    if not password:
        parser.error("Mật khẩu không được để trống.")

    settings = get_app_settings()
    store = AuthStore(resolve_path(settings.app.auth_db_path))
    try:
        store.create_user(args.username, password, display_name=args.display_name, role=args.role)
    except UserAlreadyExistsError as e:
        parser.error(str(e))
    else:
        print(f"Đã tạo tài khoản '{args.username}' ({args.role}).")
    finally:
        store.close()


if __name__ == "__main__":
    main()
