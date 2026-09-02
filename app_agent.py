# -*- coding: utf-8 -*-
"""
AI 股票问答 Agent —— 网页聊天版
运行：streamlit run app_agent.py
核心逻辑在 agent.py（函数调用循环），本文件只负责界面。
"""

import sys

# Windows 中文控制台 GBK 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

from agent import StockAgent

st.set_page_config(page_title="AI 股票问答 Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI 股票问答 Agent")
st.caption("基于 DeepSeek 函数调用 · 工具：实时行情 / K线走势摘要 / 股票代码速查")

# Agent 只创建一次（含 API 客户端），放进缓存避免重复初始化
@st.cache_resource
def get_agent() -> StockAgent:
    return StockAgent()


@st.cache_resource
def get_agent_error() -> str | None:
    """捕获 API Key 缺失等初始化错误，页面友好提示而不是白屏"""
    try:
        get_agent()
        return None
    except Exception as e:
        return str(e)


error = get_agent_error()
if error:
    st.error(f"❌ {error}")
    st.stop()

# 对话历史存在 session_state，重跑页面不丢失
st.session_state.setdefault("chat_history", [])

for m in st.session_state["chat_history"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

suggestions = ["帮我看看贵州茅台最近30天走势怎么样？",
               "比亚迪现在多少钱？市盈率多少？",
               "宁德时代和比亚迪今天谁涨得好？"]

if not st.session_state["chat_history"]:
    with st.chat_message("assistant"):
        st.write("你好！我可以查 A 股实时行情和走势。试试问我：")
        for s in suggestions:
            st.write(f"· {s}")

if prompt := st.chat_input("问我任何A股行情问题…"):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 思考中（可能正在调用行情工具）…"):
            try:
                answer = get_agent().chat(prompt)
            except Exception as e:
                answer = f"出错了：{e}"
        st.write(answer)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
