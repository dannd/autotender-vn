import pytest

from autotender.auth.store import AuthStore, UserAlreadyExistsError


def test_create_user_and_verify_correct_password(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("dan", "mat-khau-manh-123", display_name="Nguyễn Đình Đán", role="admin")

    result = store.verify_password("dan", "mat-khau-manh-123")

    assert result == {"username": "dan", "display_name": "Nguyễn Đình Đán", "role": "admin"}
    store.close()


def test_verify_wrong_password_returns_none(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("dan", "mat-khau-dung", display_name="Đán")

    assert store.verify_password("dan", "mat-khau-sai") is None
    store.close()


def test_verify_unknown_username_returns_none(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    assert store.verify_password("khong_ton_tai", "bat-ky") is None
    store.close()


def test_create_duplicate_username_raises(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("dan", "pw1", display_name="Đán")

    with pytest.raises(UserAlreadyExistsError):
        store.create_user("dan", "pw2", display_name="Đán khác")
    store.close()


def test_default_role_is_editor(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("vu", "pw", display_name="Vũ")

    result = store.verify_password("vu", "pw")

    assert result["role"] == "editor"
    store.close()


def test_list_users_does_not_expose_password_hash(tmp_path):
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("dan", "pw", display_name="Đán")

    users = store.list_users()

    assert len(users) == 1
    assert "password_hash" not in users[0]
    assert "salt" not in users[0]
    store.close()


def test_password_hash_is_salted_differently_per_user(tmp_path):
    """Cùng mật khẩu, 2 user khác nhau phải ra hash khác nhau (salt ngẫu nhiên) — chống rainbow table."""
    store = AuthStore(tmp_path / "auth.db")
    store.create_user("dan", "cung-mat-khau", display_name="Đán")
    store.create_user("vu", "cung-mat-khau", display_name="Vũ")

    row_dan = store._conn.execute("SELECT password_hash, salt FROM users WHERE username = 'dan'").fetchone()
    row_vu = store._conn.execute("SELECT password_hash, salt FROM users WHERE username = 'vu'").fetchone()

    assert row_dan["salt"] != row_vu["salt"]
    assert row_dan["password_hash"] != row_vu["password_hash"]
    store.close()
