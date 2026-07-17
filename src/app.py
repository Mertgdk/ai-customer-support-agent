import os
from dotenv import load_dotenv
import streamlit as st
from src.agent import run
from src.memory import ConversationMemory
from src.db import init_db

load_dotenv()


@st.cache_resource
def _init_db_once():
    init_db()


_init_db_once()

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
