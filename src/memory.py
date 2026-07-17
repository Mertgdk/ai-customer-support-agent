class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if role == "user" and len(self._messages) > self.max_messages:
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
