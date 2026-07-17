# AI Customer Support Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-turn AI customer support agent with tool-use and conversation memory that can look up orders, search FAQ, create tickets, and retrieve user profiles.

**Architecture:** ReAct loop (Observe->Think->Act) powered by Groq Llama-3.3-70b via LiteLLM. The agent calls tools (SQLite DB lookups, FAQ search, ticket creation) and maintains conversation history across turns using a ConversationMemory class. Streamlit provides a chat UI.

**Tech Stack:** Python 3.11+, LiteLLM (groq/llama-3.3-70b-versatile), SQLite (stdlib), Streamlit, pytest

## Global Constraints

- LLM model: `groq/llama-3.3-70b-versatile` via `litellm.completion()`
- API key env var: `GROQ_API_KEY` (never hardcoded)
- All tool return values are JSON strings (json.dumps)
- Tool schemas follow OpenAI function-calling format (type/function/name/description/parameters)
- ReAct loop max turns: `MAX_TURNS = 10`
- ConversationMemory default max_messages: 20
- DB file: `support.db` in project root (not committed)
- Run with: `python -m streamlit run src/app.py`
- Tests run with: `pytest tests/ -v`
- No .env committed — use .env.example as template

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/db.py` | SQLite init, seed data, `get_db()` |
| `src/tools.py` | 4 tool functions + `TOOL_SCHEMAS` + `TOOL_REGISTRY` |
| `src/memory.py` | `ConversationMemory` class |
| `src/agent.py` | ReAct loop `run(user_message, memory)` |
| `src/app.py` | Streamlit chat UI |
| `tests/test_tools.py` | Test each tool function |
| `tests/test_memory.py` | Test ConversationMemory |
| `tests/test_agent.py` | Test agent loop (mocked LiteLLM) |
| `requirements.txt` | Dependencies |
| `.gitignore` | Excludes .env, support.db, .venv, __pycache__ |
| `.env.example` | Template for GROQ_API_KEY |

---

### Task 1: Project Scaffolding + SQLite Database

**Files:**
- Create: `src/db.py`
- Create: `tests/__init__.py` (empty)
- Create: `src/__init__.py` (empty)
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces:
  - `get_db() -> sqlite3.Connection` (row_factory=sqlite3.Row)
  - `init_db() -> None` (idempotent - creates tables + seeds if empty)
  - `DB_PATH: Path` (project_root / "support.db")

- [ ] **Step 1: Create requirements.txt**

```
litellm>=1.40.0
streamlit>=1.35.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
.env
support.db
*.db
```

- [ ] **Step 3: Create .env.example**

```
GROQ_API_KEY=gsk_your_key_here
```

- [ ] **Step 4: Create empty __init__ files**

Create `src/__init__.py` and `tests/__init__.py` — both empty files.

- [ ] **Step 5: Write the failing test**

Create `tests/test_db.py`:

```python
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
```

- [ ] **Step 6: Run test — expect FAIL**

```
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.db'`

- [ ] **Step 7: Create src/db.py**

```python
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "support.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


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
```

- [ ] **Step 8: Install dependencies**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- [ ] **Step 9: Run test — expect PASS**

```
pytest tests/test_db.py -v
```

Expected: `3 passed`

- [ ] **Step 10: Commit**

```bash
git add src/__init__.py src/db.py tests/__init__.py tests/test_db.py requirements.txt .gitignore .env.example
git commit -m "feat: project scaffold + SQLite DB with seed data"
```

---

### Task 2: Tool Functions

**Files:**
- Create: `src/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `get_db() -> sqlite3.Connection`, `init_db() -> None` from `src.db`
- Produces:
  - `lookup_order(order_id: str) -> str` (JSON string)
  - `search_faq(query: str) -> str` (JSON string)
  - `create_ticket(email: str, issue: str) -> str` (JSON string)
  - `get_user_info(email: str) -> str` (JSON string)
  - `TOOL_SCHEMAS: list[dict]` (OpenAI function-calling format)
  - `TOOL_REGISTRY: dict[str, callable]` (name -> function)

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools.py`:

```python
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
```

- [ ] **Step 2: Run test — expect FAIL**

```
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.tools'`

- [ ] **Step 3: Create src/tools.py**

```python
import json
from datetime import datetime
from src.db import get_db, init_db

FAQ = [
    {"q": "How do I return a product?", "a": "Returns accepted within 30 days of delivery. Start at returns.example.com or email support@example.com with your order ID."},
    {"q": "What is the shipping time?", "a": "Standard: 3-5 business days. Express: 1-2 business days. Free shipping on orders over $50."},
    {"q": "How do I track my order?", "a": "Use the tracking number from your shipping confirmation at track.example.com, or ask me to look up your order ID."},
    {"q": "What is the refund policy?", "a": "Refunds processed within 5-7 business days after we receive the item. Original shipping fees are non-refundable."},
    {"q": "How do I cancel my order?", "a": "Orders can be cancelled within 1 hour of placement. After that, wait for delivery then initiate a return."},
    {"q": "What payment methods do you accept?", "a": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."},
    {"q": "Is my data safe?", "a": "Yes. We use 256-bit encryption and never share your data with third parties. See privacy policy at example.com/privacy."},
    {"q": "How do I contact support?", "a": "Email support@example.com, call 1-800-EXAMPLE (Mon-Fri 9am-6pm EST), or use live chat on our website."},
]


def lookup_order(order_id: str) -> str:
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, customer_email, product, status, tracking_number, created_at FROM orders WHERE id = ?",
            (order_id.upper(),),
        ).fetchone()
    if row is None:
        return json.dumps({"error": f"Order {order_id} not found."})
    return json.dumps({
        "order_id": row["id"],
        "customer_email": row["customer_email"],
        "product": row["product"],
        "status": row["status"],
        "tracking_number": row["tracking_number"],
        "created_at": row["created_at"],
    })


def search_faq(query: str) -> str:
    query_words = set(query.lower().split())
    scored = []
    for entry in FAQ:
        text = (entry["q"] + " " + entry["a"]).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [e for _, e in scored[:3]]
    if not results:
        return json.dumps({"results": [], "note": "No FAQ match found."})
    return json.dumps({"results": results})


def create_ticket(email: str, issue: str) -> str:
    init_db()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO tickets (email, issue, status, created_at) VALUES (?, ?, 'open', ?)",
            (email, issue, datetime.now().isoformat()),
        )
        ticket_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return json.dumps({
        "ticket_id": ticket_id,
        "status": "open",
        "message": f"Ticket #{ticket_id} created. Our team will respond within 24 hours.",
    })


def get_user_info(email: str) -> str:
    init_db()
    with get_db() as conn:
        row = conn.execute(
            "SELECT email, name, tier FROM users WHERE email = ?",
            (email.lower(),),
        ).fetchone()
    if row is None:
        return json.dumps({"error": f"User {email} not found."})
    return json.dumps({"email": row["email"], "name": row["name"], "tier": row["tier"]})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its ID (e.g. ORD-001). Returns product, status, tracking number, and customer email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. ORD-001"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search the FAQ for answers to common questions about shipping, returns, refunds, cancellations, and payment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The customer question or keywords to search"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket for issues that cannot be resolved immediately. Use when the customer needs escalation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Customer email address"},
                    "issue": {"type": "string", "description": "Detailed description of the issue"},
                },
                "required": ["email", "issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Retrieve customer profile (name, account tier) by email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Customer email address"}
                },
                "required": ["email"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "lookup_order": lookup_order,
    "search_faq": search_faq,
    "create_ticket": create_ticket,
    "get_user_info": get_user_info,
}
```

- [ ] **Step 4: Run test — expect PASS**

```
pytest tests/test_tools.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/tools.py tests/test_tools.py
git commit -m "feat: tool functions (lookup_order, search_faq, create_ticket, get_user_info)"
```

---

### Task 3: Conversation Memory

**Files:**
- Create: `src/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces:
  - `class ConversationMemory`
    - `__init__(self, max_messages: int = 20)`
    - `add(self, role: str, content: str) -> None`
    - `add_assistant_tool_calls(self, raw_tool_calls: list[dict]) -> None`
    - `add_tool_result(self, tool_call_id: str, content: str) -> None`
    - `get_messages(self) -> list[dict]`
    - `reset(self) -> None`

- [ ] **Step 1: Write failing tests**

Create `tests/test_memory.py`:

```python
from src.memory import ConversationMemory


def test_add_user_message():
    mem = ConversationMemory()
    mem.add("user", "Hello")
    msgs = mem.get_messages()
    assert msgs == [{"role": "user", "content": "Hello"}]


def test_add_multiple_messages():
    mem = ConversationMemory()
    mem.add("user", "Hello")
    mem.add("assistant", "Hi!")
    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_reset_clears_all_messages():
    mem = ConversationMemory()
    mem.add("user", "Hello")
    mem.add("assistant", "Hi!")
    mem.reset()
    assert mem.get_messages() == []


def test_max_messages_trims_oldest():
    mem = ConversationMemory(max_messages=3)
    mem.add("user", "msg1")
    mem.add("assistant", "msg2")
    mem.add("user", "msg3")
    mem.add("assistant", "msg4")
    msgs = mem.get_messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg2"
    assert msgs[2]["content"] == "msg4"


def test_get_messages_returns_copy():
    mem = ConversationMemory()
    mem.add("user", "Hello")
    msgs = mem.get_messages()
    msgs.append({"role": "user", "content": "injected"})
    assert len(mem.get_messages()) == 1


def test_add_assistant_tool_calls():
    mem = ConversationMemory()
    raw_calls = [{"id": "tc1", "type": "function", "function": {"name": "lookup_order", "arguments": '{"order_id": "ORD-001"}'}}]
    mem.add_assistant_tool_calls(raw_calls)
    msgs = mem.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] is None
    assert msgs[0]["tool_calls"] == raw_calls


def test_add_tool_result():
    mem = ConversationMemory()
    mem.add_tool_result("tc1", '{"status": "delivered"}')
    msgs = mem.get_messages()
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "tc1"
    assert msgs[0]["content"] == '{"status": "delivered"}'
```

- [ ] **Step 2: Run test — expect FAIL**

```
pytest tests/test_memory.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.memory'`

- [ ] **Step 3: Create src/memory.py**

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_assistant_tool_calls(self, raw_tool_calls: list[dict]) -> None:
        self._messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": raw_tool_calls,
        })

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()
```

- [ ] **Step 4: Run test — expect PASS**

```
pytest tests/test_memory.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/memory.py tests/test_memory.py
git commit -m "feat: ConversationMemory with max_messages trimming"
```

---

### Task 4: ReAct Agent Loop

**Files:**
- Create: `src/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes:
  - `litellm.completion(model, messages, tools, tool_choice)` from litellm
  - `TOOL_SCHEMAS: list[dict]` from `src.tools`
  - `TOOL_REGISTRY: dict[str, callable]` from `src.tools`
  - `ConversationMemory` from `src.memory`
- Produces:
  - `SYSTEM_PROMPT: str`
  - `MAX_TURNS: int = 10`
  - `run(user_message: str, memory: ConversationMemory) -> str`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
import json
from unittest.mock import patch, MagicMock
from src.agent import run
from src.memory import ConversationMemory


def _text_response(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_response(tool_name: str, tool_id: str, tool_args: str):
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = tool_name
    tc.function.arguments = tool_args
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_run_returns_string():
    with patch("src.agent.litellm.completion", return_value=_text_response("Hello!")):
        mem = ConversationMemory()
        result = run("Hi", mem)
    assert isinstance(result, str)
    assert result == "Hello!"


def test_run_adds_user_and_assistant_to_memory():
    with patch("src.agent.litellm.completion", return_value=_text_response("I can help.")):
        mem = ConversationMemory()
        run("Help me", mem)
    msgs = mem.get_messages()
    assert msgs[0] == {"role": "user", "content": "Help me"}
    assert msgs[1] == {"role": "assistant", "content": "I can help."}


def test_run_calls_tool_then_returns_final_answer():
    tool_resp = _tool_response("lookup_order", "tc1", '{"order_id": "ORD-001"}')
    final_resp = _text_response("Your order ORD-001 is delivered.")

    fake_registry = {
        "lookup_order": lambda order_id: json.dumps({"status": "delivered", "product": "Laptop"})
    }

    with patch("src.agent.litellm.completion", side_effect=[tool_resp, final_resp]):
        with patch("src.agent.TOOL_REGISTRY", fake_registry):
            mem = ConversationMemory()
            result = run("Where is ORD-001?", mem)

    assert result == "Your order ORD-001 is delivered."
    msgs = mem.get_messages()
    # user + assistant(tool_call) + tool_result + assistant(final)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"] is not None
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "tc1"
    assert msgs[3]["role"] == "assistant"
    assert msgs[3]["content"] == "Your order ORD-001 is delivered."


def test_run_unknown_tool_returns_error_observation():
    tool_resp = _tool_response("nonexistent_tool", "tc2", '{}')
    final_resp = _text_response("Sorry, something went wrong.")

    with patch("src.agent.litellm.completion", side_effect=[tool_resp, final_resp]):
        with patch("src.agent.TOOL_REGISTRY", {}):
            mem = ConversationMemory()
            result = run("Do something", mem)

    msgs = mem.get_messages()
    tool_result_msg = next(m for m in msgs if m["role"] == "tool")
    tool_result_data = json.loads(tool_result_msg["content"])
    assert "error" in tool_result_data


def test_run_includes_system_prompt_in_llm_call():
    from src.agent import SYSTEM_PROMPT
    captured_calls = []

    def capture_call(**kwargs):
        captured_calls.append(kwargs)
        return _text_response("OK")

    with patch("src.agent.litellm.completion", side_effect=capture_call):
        mem = ConversationMemory()
        run("Hi", mem)

    messages_sent = captured_calls[0]["messages"]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[0]["content"] == SYSTEM_PROMPT
```

- [ ] **Step 2: Run test — expect FAIL**

```
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agent'`

- [ ] **Step 3: Create src/agent.py**

```python
import json
import litellm
from src.tools import TOOL_SCHEMAS, TOOL_REGISTRY
from src.memory import ConversationMemory

SYSTEM_PROMPT = """You are a helpful customer support agent for ExampleStore.
You help customers with orders, returns, shipping, refunds, and general questions.
Always be polite and professional. Use the available tools to look up real information.
Never invent order details — always use the lookup_order tool for order status.
If you cannot resolve an issue, create a support ticket using the create_ticket tool."""

MAX_TURNS = 10


def run(user_message: str, memory: ConversationMemory) -> str:
    memory.add("user", user_message)

    for _ in range(MAX_TURNS):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get_messages()

        response = litellm.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        if not tool_calls:
            answer = msg.content
            memory.add("assistant", answer)
            return answer

        raw_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
        memory.add_assistant_tool_calls(raw_calls)

        for tc in tool_calls:
            fn = TOOL_REGISTRY.get(tc.function.name)
            if fn is None:
                result = json.dumps({"error": f"Unknown tool: {tc.function.name}"})
            else:
                try:
                    args = json.loads(tc.function.arguments)
                    result = fn(**args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})
            memory.add_tool_result(tc.id, result)

    return "I was unable to resolve your issue. Please contact support@example.com directly."
```

- [ ] **Step 4: Run test — expect PASS**

```
pytest tests/test_agent.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass (test_db + test_tools + test_memory + test_agent)

- [ ] **Step 6: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: ReAct agent loop with tool-use and conversation memory"
```

---

### Task 5: Streamlit Chat UI

**Files:**
- Create: `src/app.py`
- No automated test (manual UI verification)

**Interfaces:**
- Consumes:
  - `run(user_message: str, memory: ConversationMemory) -> str` from `src.agent`
  - `ConversationMemory` from `src.memory`
  - `init_db()` from `src.db`

- [ ] **Step 1: Create src/app.py**

```python
import os
from dotenv import load_dotenv
import streamlit as st
from src.agent import run
from src.memory import ConversationMemory
from src.db import init_db

load_dotenv()
init_db()

st.set_page_config(page_title="ExampleStore Support", layout="centered")
st.title("ExampleStore Customer Support")
st.caption("Ask about your orders, returns, shipping, and more.")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run(prompt, st.session_state.memory)
        st.write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("Test Data")
    st.markdown("""
**Test orders:**
- `ORD-001` - Alice, Laptop Pro X1, **delivered**
- `ORD-002` - Alice, Headphones, **shipped** (TRK-789012)
- `ORD-003` - Bob, USB-C Hub, **processing**
- `ORD-004` - Carol, Monitor, **cancelled**

**Test emails:**
- `alice@example.com` (premium)
- `bob@example.com` (free)
- `carol@example.com` (enterprise)

**Try asking:**
- "Where is my order ORD-002?"
- "How do I return a product?"
- "I have a problem with my order, my email is bob@example.com"
    """)
    if st.button("Clear conversation"):
        st.session_state.memory.reset()
        st.session_state.chat_history = []
        st.rerun()
```

- [ ] **Step 2: Set up .env**

Copy `.env.example` to `.env` and fill in the Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

- [ ] **Step 3: Run the app**

```
python -m streamlit run src/app.py
```

- [ ] **Step 4: Manual verification**

Test these scenarios in order:
1. Ask "Where is my order ORD-001?" - agent should call `lookup_order` and return status
2. Ask "How do I return it?" - agent should call `search_faq` and explain return policy
3. Ask "My email is bob@example.com, I have a problem with ORD-003" - agent should call `get_user_info` then `create_ticket`
4. Click "Clear conversation" and verify chat resets
5. Ask "What payment methods do you accept?" - agent should call `search_faq`

- [ ] **Step 5: Commit**

```bash
git add src/app.py
git commit -m "feat: Streamlit chat UI with sidebar test data"
```

---

### Task 6: GitHub Push + Copy to Project Folder

- [ ] **Step 1: Create GitHub repo**

Go to https://github.com/new and create `ai-customer-support-agent` (public).

- [ ] **Step 2: Add remote and push**

```bash
git remote add origin https://github.com/Mertgdk/ai-customer-support-agent.git
git branch -M main
git push -u origin main
```

- [ ] **Step 3: Copy to project folder**

```
xcopy /E /I /Y "C:\Users\MERT\Desktop\ai-customer-support-agent" "D:\mertt\Masaüstü\PROJE_CALISMALARIM\AI Engineer\ai-customer-support-agent"
```

- [ ] **Step 4: Update memory**

Update `project_ai_engineering.md` — mark Project 2 as DONE.
