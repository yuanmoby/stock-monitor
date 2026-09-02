# -*- coding: utf-8 -*-
"""
AI 股票问答 Agent —— 基于大模型函数调用（Function Calling）的智能体

工作原理（面试必讲）：
    模型不再只靠"记忆"回答，而是拥有一个工具清单。用户提问后，
    模型判断"我需要查数据"，返回一个工具调用请求（函数名+参数），
    程序执行工具拿到真实数据，把结果回传给模型，模型基于数据
    生成最终回答。这个"提问 → 调用工具 → 拿结果 → 再回答"的
    循环就是 Agent 的核心。

技术要点：
    1. 直接对接 DeepSeek 的 OpenAI 兼容 HTTP 接口（不依赖 SDK），
       协议：POST /chat/completions，body 里带 messages + tools
    2. 工具执行复用股票监测项目的 data_source 数据层
    3. 工具失败时把错误文本回传给模型，让模型自动换思路，而不是
       让整个 Agent 崩溃（容错设计）
    4. 最大 5 轮工具调用，防止死循环

运行：
    python agent.py "茅台最近30天走势怎么样？"
网页版：
    streamlit run app_agent.py
需要 DeepSeek API Key：https://platform.deepseek.com 注册后在
项目目录新建 .env 文件（不会提交到 Git）：
    DEEPSEEK_API_KEY=sk-xxxx
"""

import json
import os
import sys
from pathlib import Path

import requests

# Windows 中文控制台 GBK 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_source import get_quote, load_kline  # noqa: E402

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
MAX_TOOL_ROUNDS = 5  # 最多调用几轮工具，防止死循环

# ------------------------------------------------------------
# 股票名称 → 代码 速查表（静态表：够演示用，不依赖网络）
# ------------------------------------------------------------
STOCK_NAMES = {
    "贵州茅台": "600519", "五粮液": "000858", "泸州老窖": "000568",
    "中国平安": "601318", "招商银行": "600036", "工商银行": "601398",
    "平安银行": "000001", "万科A": "000002", "保利发展": "600048",
    "比亚迪": "002594", "宁德时代": "300750", "长城汽车": "601633",
    "中芯国际": "688981", "中兴通讯": "000063", "寒武纪": "688256",
    "科大讯飞": "002230", "海康威视": "002415", "立讯精密": "002475",
    "长江电力": "600900", "中国石油": "601857", "中国神华": "601088",
    "隆基绿能": "601012", "恒瑞医药": "600276", "迈瑞医疗": "300760",
    "东方财富": "300059", "中信证券": "600030", "格力电器": "000651",
    "美的集团": "000333", "紫金矿业": "601899", "药明康德": "603259",
}


def _load_api_key() -> str | None:
    """优先读环境变量 DEEPSEEK_API_KEY，其次读项目根目录 .env 文件"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ------------------------------------------------------------
# 工具定义：告诉模型"有哪些工具、参数怎么传"（JSON Schema）
# ------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "查询一只A股股票的实时行情：最新价、涨跌幅、今开、最高、最低、昨收、成交额、换手率、市盈率",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string", "description": "6位数字股票代码，例如 600519"}
                },
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kline_summary",
            "description": "查询一只股票最近N个交易日的走势摘要：区间涨跌幅、区间最高价、区间最低价、最新收盘价",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string", "description": "6位数字股票代码，例如 600519"},
                    "days": {"type": "integer", "description": "最近多少个交易日，例如 30"},
                },
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_stock_code",
            "description": "根据股票名称关键词查找6位股票代码，例如输入'茅台'返回贵州茅台的代码。用户只说了名称、没说代码时，必须先调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "股票名称关键词"}
                },
                "required": ["keyword"],
            },
        },
    },
]


# ------------------------------------------------------------
# 工具实现（真正的执行逻辑，复用 data_source 数据层）
# ------------------------------------------------------------

def _tool_find_stock_code(keyword: str) -> str:
    hits = [f"{name}({code})" for name, code in STOCK_NAMES.items() if keyword in name]
    if not hits:
        return f"未找到名称包含「{keyword}」的股票，请换个关键词"
    return "匹配到：" + "、".join(hits)


def _tool_get_stock_quote(stock_code: str) -> str:
    try:
        q, from_cache = get_quote(stock_code.strip())
        if q is None:
            return f"未找到股票代码 {stock_code}，请检查代码是否正确"
        src = "本地缓存数据（接口暂不可用，注意不是实时价）" if from_cache else "实时数据"
        pe = f"{q['市盈率']:.2f}" if q["市盈率"] else "N/A"
        return (
            f"{q['名称']}（{stock_code}）行情，{src}：最新价 {q['最新价']:.2f} 元，"
            f"涨跌幅 {q['涨跌幅']:+.2f}%，今开 {q['今开']:.2f}，最高 {q['最高']:.2f}，"
            f"最低 {q['最低']:.2f}，昨收 {q['昨收']:.2f}，成交额 {q['成交额']/1e8:.2f} 亿元，"
            f"换手率 {q['换手率']:.2f}%，市盈率 {pe}"
        )
    except Exception as e:
        return f"查询失败：{e}"


def _tool_get_kline_summary(stock_code: str, days: int = 30) -> str:
    try:
        df, from_cache = load_kline(stock_code.strip(), days=int(days))
        first_close = df.iloc[0]["收盘"]
        last = df.iloc[-1]
        chg = (last["收盘"] - first_close) / first_close * 100
        src = "缓存数据" if from_cache else "实时数据"
        return (
            f"{stock_code} 最近 {len(df)} 个交易日走势摘要（{src}）："
            f"最新收盘 {last['收盘']:.2f} 元（{last['日期']}），区间涨跌幅 {chg:+.2f}%，"
            f"区间最高 {df['最高'].max():.2f}，区间最低 {df['最低'].min():.2f}"
        )
    except Exception as e:
        return f"查询失败：{e}"


TOOL_MAP = {
    "find_stock_code": _tool_find_stock_code,
    "get_stock_quote": _tool_get_stock_quote,
    "get_kline_summary": _tool_get_kline_summary,
}


# ------------------------------------------------------------
# Agent 主类：函数调用循环
# ------------------------------------------------------------

SYSTEM_PROMPT = """你是一个A股行情问答助手，可以调用工具查询真实行情数据。规则：
1. 用户只说了股票名称、没给代码时，必须先用 find_stock_code 查代码，再查行情。
2. 回答要给出关键数字和单位（元、亿元、%），并说明数据是实时还是缓存。
3. 严禁编造数据：工具没有返回的信息，就说查不到。
4. 数据若来自缓存（接口暂不可用时），要提醒用户"这是缓存数据，不是实时价"。
5. 回答简洁，用中文。"""


class StockAgent:
    """股票问答 Agent：模型决策 → 工具执行 → 结果回传 → 生成回答"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            raise RuntimeError(
                "未找到 DeepSeek API Key。请到 https://platform.deepseek.com 注册并创建 Key，"
                "然后在项目目录新建 .env 文件写入：DEEPSEEK_API_KEY=sk-xxxx"
            )

    def _call_llm(self, messages: list[dict]) -> dict:
        """调 DeepSeek 接口（OpenAI 兼容协议，requests 直连不依赖 SDK）"""
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "temperature": 0.3,  # 低温度：行情问答要稳定，不要创意
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def _run_tool(self, name: str, args: dict) -> str:
        """执行工具，失败时返回错误文本让模型自己换思路（容错设计）"""
        fn = TOOL_MAP.get(name)
        if fn is None:
            return f"未知工具：{name}"
        try:
            return fn(**args)
        except TypeError:
            # 模型传参不规范时兜底
            return f"工具 {name} 参数错误：{args}"
        except Exception as e:
            return f"工具 {name} 执行失败：{e}"

    def chat(self, user_input: str, verbose: bool = False) -> str:
        """Agent 主循环：最多 MAX_TOOL_ROUNDS 轮工具调用"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        for _ in range(MAX_TOOL_ROUNDS):
            data = self._call_llm(messages)
            msg = data["choices"][0]["message"]

            # 模型没有请求工具 → 直接返回最终回答
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg.get("content") or "（模型未返回内容）"

            # 把模型的工具调用消息原样放回历史，再逐个执行并追加结果
            messages.append(msg)
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(name, args)
                if verbose:
                    print(f"  🔧 调用工具 {name}{args}")
                    print(f"     → {result[:90]}")
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )

        return "（已达到最大工具调用轮数，仍未获得最终回答，请换个问法）"


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "帮我看看贵州茅台最近30天走势怎么样？"
    agent = StockAgent()
    print(f"🙋 提问：{question}\n")
    answer = agent.chat(question, verbose=True)
    print(f"\n🤖 回答：\n{answer}")


if __name__ == "__main__":
    main()
