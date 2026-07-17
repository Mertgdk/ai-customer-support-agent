import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    import src.db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")


def test_init_db_creates_tables():
    from src.db import init_db, get_db
    init_db()
    with get_db() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"users", "orders", "tickets"}.issubset(tables)


def test_init_db_seeds_data():
    from src.db import init_db, get_db
    init_db()
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert user_count == 3
    assert order_count == 4


def test_init_db_idempotent():
    from src.db import init_db, get_db
    init_db()
    init_db()  # second call should not duplicate data
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert user_count == 3
