import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "support.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'free'
            );
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                customer_email TEXT NOT NULL,
                product TEXT NOT NULL,
                status TEXT NOT NULL,
                tracking_number TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                issue TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            );
        """)
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO users (email, name, tier) VALUES (?, ?, ?)",
                [
                    ("alice@example.com", "Alice Chen", "premium"),
                    ("bob@example.com", "Bob Smith", "free"),
                    ("carol@example.com", "Carol White", "enterprise"),
                ],
            )
            conn.executemany(
                "INSERT INTO orders (id, customer_email, product, status, tracking_number, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("ORD-001", "alice@example.com", "Laptop Pro X1", "delivered", "TRK-123456", "2026-07-01"),
                    ("ORD-002", "alice@example.com", "Wireless Headphones", "shipped", "TRK-789012", "2026-07-10"),
                    ("ORD-003", "bob@example.com", "USB-C Hub", "processing", None, "2026-07-15"),
                    ("ORD-004", "carol@example.com", "Monitor 27inch", "cancelled", None, "2026-07-05"),
                ],
            )
