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
