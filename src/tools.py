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
