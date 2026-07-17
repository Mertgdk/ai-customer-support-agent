import json
import pytest


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    import src.db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    from src.db import init_db
    init_db()


def test_lookup_order_found():
    from src.tools import lookup_order
    data = json.loads(lookup_order("ORD-001"))
    assert data["order_id"] == "ORD-001"
    assert data["status"] == "delivered"
    assert data["product"] == "Laptop Pro X1"
    assert data["tracking_number"] == "TRK-123456"


def test_lookup_order_case_insensitive():
    from src.tools import lookup_order
    data = json.loads(lookup_order("ord-001"))
    assert data["order_id"] == "ORD-001"


def test_lookup_order_not_found():
    from src.tools import lookup_order
    data = json.loads(lookup_order("ORD-999"))
    assert "error" in data


def test_search_faq_returns_relevant_result():
    from src.tools import search_faq
    data = json.loads(search_faq("How do I return a product?"))
    assert len(data["results"]) > 0
    assert any("return" in r["q"].lower() for r in data["results"])


def test_search_faq_no_match():
    from src.tools import search_faq
    data = json.loads(search_faq("xyzqwerty"))
    assert data["results"] == []


def test_create_ticket_returns_ticket_id():
    from src.tools import create_ticket
    data = json.loads(create_ticket("test@example.com", "My order is missing"))
    assert "ticket_id" in data
    assert data["status"] == "open"


def test_create_ticket_persisted_in_db():
    from src.tools import create_ticket
    from src.db import get_db
    create_ticket("test@example.com", "Missing item")
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE email = ?", ("test@example.com",)).fetchone()
    assert row is not None
    assert row["issue"] == "Missing item"


def test_get_user_info_found():
    from src.tools import get_user_info
    data = json.loads(get_user_info("alice@example.com"))
    assert data["name"] == "Alice Chen"
    assert data["tier"] == "premium"


def test_get_user_info_not_found():
    from src.tools import get_user_info
    data = json.loads(get_user_info("nobody@example.com"))
    assert "error" in data


def test_tool_schemas_structure():
    from src.tools import TOOL_SCHEMAS
    assert len(TOOL_SCHEMAS) == 4
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_tool_registry_has_all_tools():
    from src.tools import TOOL_REGISTRY
    assert set(TOOL_REGISTRY.keys()) == {"lookup_order", "search_faq", "create_ticket", "get_user_info"}
