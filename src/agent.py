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
            answer = msg.content or ""
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
