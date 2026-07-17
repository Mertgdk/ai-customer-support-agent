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
    # Trim only fires on user messages. With max_messages=3:
    # add user msg1 (1), add assistant msg2 (2), add user msg3 (3),
    # add user msg4 (4 > 3) -> trim to last 3: [msg2, msg3, msg4]
    mem = ConversationMemory(max_messages=3)
    mem.add("user", "msg1")
    mem.add("assistant", "msg2")
    mem.add("user", "msg3")
    mem.add("user", "msg4")
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
