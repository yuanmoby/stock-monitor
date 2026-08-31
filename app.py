"""
实时股票行情监测工具
基于 Streamlit + Akshare 搭建的 A 股实时行情查询应用
作者：齐北
日期：2025.06 初版 / 2026.08 重构
"""

import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# 数据获取逻辑在 data_source.py，与 lstm_predict.py 共用
from data_source import get_quote, load_kline

st.set_page_config(page_title="实时股票行情监测", page_icon="📈", layout="wide")

# ============================================================
# 第1部分：数据层（内存缓存）
# ============================================================
# 内存缓存是 Streamlit 层面的第二道缓存：
# - 报价缓存 30 秒：单股接口一次只拉一只股票，30 秒足够兼顾新鲜度和请求量
# - K线缓存 5 分钟：K 线按日更新，盘中最多只影响最后一根
# 数据源本身的"本地磁盘缓存降级"在 data_source.py 里实现。

@st.cache_data(ttl=30, show_spinner=False)
def cached_quote(stock_code: str):
    """实时报价，30 秒内存缓存"""
    return get_quote(stock_code)


@st.cache_data(ttl=300)
def cached_kline(stock_code: str, days: int = 60):
    """历史K线，5 分钟内存缓存"""
    return load_kline(stock_code, days)


# ============================================================
# 第2部分：页面输入区
# ============================================================

st.title("📈 实时股票行情监测工具")
st.caption("数据来源：东方财富 | 更新频率：实时 | A股市场")

# 默认展示的股票
st.session_state.setdefault("stock_code", "600900")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.text_input(
        "请输入股票代码（6位数字）",
        key="stock_code",
        max_chars=6,
        placeholder="例如：000001（平安银行）",
    )

with col2:
    st.write("")
    st.write("")
    query_btn = st.button("🔍 查询行情", type="primary", width='stretch')

with col3:
    st.write("")
    st.write("")
    refresh_btn = st.button("🔄 刷新数据", width='stretch')

# 常用股票快速选择：点击后自动填入输入框并加载
with st.expander("📋 常用股票快速选择"):
    common_stocks = {
        "平安银行": "000001", "万科A": "000002", "中国平安": "601318",
        "贵州茅台": "600519", "比亚迪": "002594", "宁德时代": "300750",
        "招商银行": "600036", "五粮液": "000858", "中芯国际": "688981",
        "中兴通讯": "000063", "科大讯飞": "002230", "寒武纪": "688256",
    }
    cols = st.columns(6)
    for i, (name, code) in enumerate(common_stocks.items()):
        with cols[i % 6]:
            if st.button(f"{code} {name}", key=f"pick_{code}", width='stretch'):
                st.session_state["stock_code"] = code


# ============================================================
# 第3部分：数据展示
# ============================================================

def show_price_card(data: dict):
    """展示股价指标卡片"""
    change_sign = "+" if data["涨跌额"] >= 0 else ""

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📌 最新价", f"¥{data['最新价']:.2f}")
    with col2:
        st.metric(
            "📊 涨跌额",
            f"{change_sign}{data['涨跌额']:.2f}",
            delta=f"{change_sign}{data['涨跌幅']:.2f}%",
        )
    with col3:
        st.metric("🏁 今开", f"¥{data['今开']:.2f}")
    with col4:
        st.metric("⬆ 最高", f"¥{data['最高']:.2f}")
    with col5:
        st.metric("⬇ 最低", f"¥{data['最低']:.2f}")
    with col6:
        st.metric("📋 昨收", f"¥{data['昨收']:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 成交额", f"{data['成交额']/1e8:.2f}亿")
    with col2:
        st.metric("📦 成交量", f"{data['成交量']/1e4:.0f}万手")
    with col3:
        st.metric("🔄 换手率", f"{data['换手率']:.2f}%")
    with col4:
        pe_val = f"{data['市盈率']:.2f}" if data["市盈率"] else "N/A"
        st.metric("📐 市盈率", pe_val)


def show_kline_chart(kline_df: pd.DataFrame):
    """K 线图 + MA5/MA10 均线 + 成交额柱状图"""
    kline_df = kline_df.copy()
    # 均线：对收盘价做滚动平均，辅助观察短期趋势
    kline_df["MA5"] = kline_df["收盘"].rolling(5).mean()
    kline_df["MA10"] = kline_df["收盘"].rolling(10).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("K线图（前复权，含MA5/MA10）", "成交额"),
    )

    # ---- K线（红涨绿跌，A股习惯）----
    fig.add_trace(
        go.Candlestick(
            x=kline_df["日期"],
            open=kline_df["开盘"],
            high=kline_df["最高"],
            low=kline_df["最低"],
            close=kline_df["收盘"],
            name="K线",
            increasing_line_color="#ff4d4f",
            decreasing_line_color="#52c41a",
        ),
        row=1, col=1,
    )

    # ---- 均线 ----
    fig.add_trace(
        go.Scatter(x=kline_df["日期"], y=kline_df["MA5"],
                   name="MA5", line=dict(color="#faad14", width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=kline_df["日期"], y=kline_df["MA10"],
                   name="MA10", line=dict(color="#1677ff", width=1.2)),
        row=1, col=1,
    )

    # ---- 成交额柱状图：涨红跌绿 ----
    colors = [
        "#ff4d4f" if close >= open_ else "#52c41a"
        for close, open_ in zip(kline_df["收盘"], kline_df["开盘"])
    ]
    fig.add_trace(
        go.Bar(x=kline_df["日期"], y=kline_df["成交额"],
               name="成交额", marker_color=colors, opacity=0.6),
        row=2, col=1,
    )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="价格（元）", row=1, col=1)
    fig.update_yaxes(title_text="成交额（元）", row=2, col=1)

    st.plotly_chart(fig, width='stretch')


# ============================================================
# 第4部分：主流程
# ============================================================

st.session_state.setdefault("shown_code", None)
st.session_state.setdefault("last_data", None)
st.session_state.setdefault("last_kline", None)

code = str(st.session_state["stock_code"]).strip()
valid = bool(re.fullmatch(r"\d{6}", code))

# 需要加载的条件：点了按钮，或输入框切换到了一个新的有效代码
need_load = query_btn or refresh_btn or st.session_state["shown_code"] != code

if not valid:
    st.info("💡 请输入6位数字的股票代码，例如：000001（平安银行）")
elif need_load:
    with st.spinner("⏳ 正在获取行情数据..."):
        data, quote_from_cache = cached_quote(code)
        try:
            kline, kline_from_cache = cached_kline(code)
        except Exception as e:
            st.error(f"获取K线数据失败：{e}")
            kline, kline_from_cache = None, False

    if data is None:
        st.warning(f"❌ 未找到股票代码 {code}，请检查后重试")
        st.session_state["shown_code"] = None
        st.session_state["last_data"] = None
    else:
        st.session_state["shown_code"] = code
        st.session_state["last_data"] = data
        st.session_state["last_kline"] = kline
        st.session_state["last_flags"] = (quote_from_cache, kline_from_cache)

# 展示已加载的数据
if valid and st.session_state["shown_code"] == code and st.session_state["last_data"] is not None:
    data = st.session_state["last_data"]
    kline = st.session_state["last_kline"]
    quote_from_cache, kline_from_cache = st.session_state.get("last_flags", (False, False))

    if quote_from_cache:
        st.warning("⚠️ 实时接口暂不可用，报价为本地缓存数据（上次成功拉取）")
    if kline_from_cache:
        st.warning("⚠️ K线接口暂不可用，K线为本地缓存数据（上次成功拉取）")
    if not quote_from_cache and not kline_from_cache:
        st.success(
            f"✅ {data['名称']}（{code}）行情加载成功 | 更新时间：{datetime.now().strftime('%H:%M:%S')}"
        )

    st.subheader(f"📊 {data['名称']} — 实时行情")
    show_price_card(data)

    st.subheader("📈 K线走势（近60个交易日）")
    if kline is not None and not kline.empty:
        show_kline_chart(kline)
        with st.expander("📋 查看原始数据"):
            st.dataframe(kline.tail(20), width='stretch')
    else:
        st.info("暂无K线数据")

# 页脚
st.divider()
st.caption("⚠️ 声明：本工具仅用于学习与技术展示，不构成任何投资建议。股市有风险，投资需谨慎。")
