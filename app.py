### 4.0 完整代码（先看一遍，再逐段理解）


"""
实时股票行情监测工具
基于 Streamlit + Akshare 搭建的 A 股实时行情查询应用
作者：齐北
日期：2025.06
"""

import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# 第1部分：页面基础配置
# ============================================================
st.set_page_config(
    page_title="实时股票行情监测",
    page_icon="📈",
    layout="wide",
)

st.title("📈 实时股票行情监测工具")
st.caption("数据来源：东方财富 | 更新频率：实时 | A股市场")

# ============================================================
# 第2部分：用户输入区
# ============================================================
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    stock_code = st.text_input(
        "请输入股票代码（6位数字）",
        value="600900",
        max_chars=6,
        placeholder="例如：600900（长江电力）",
    )

with col2:
    st.write("")  # 占位对齐
    st.write("")
    query_btn = st.button("🔍 查询行情", type="primary", use_container_width=True)

with col3:
    st.write("")
    st.write("")
    refresh_btn = st.button("🔄 刷新数据", use_container_width=True)

# ============================================================
# 第3部分：获取实时行情数据
# ============================================================
@st.cache_data(ttl=30)  # 缓存30秒，避免频繁请求
def get_realtime_price(stock_code: str):
    """
    获取单只股票的实时行情
    参数:
        stock_code: 6位股票代码，如 '000001'
    返回:
        dict: 包含股票名称和各项行情数据的字典
    """
    try:
        # Akshare接口：获取A股实时行情（全部股票）
        df = ak.stock_zh_a_spot_em()

        # 按股票代码筛选
        df_filtered = df[df["代码"] == stock_code]

        if df_filtered.empty:
            return None

        row = df_filtered.iloc[0]
        return {
            "名称": row["名称"],
            "最新价": float(row["最新价"]),
            "涨跌额": float(row["涨跌额"]),
            "涨跌幅": float(row["涨跌幅"]),
            "今开": float(row["今开"]),
            "最高": float(row["最高"]),
            "最低": float(row["最低"]),
            "昨收": float(row["昨收"]),
            "成交量": int(row["成交量"]),
            "成交额": float(row["成交额"]),
            "换手率": float(row["换手率"]),
            "市盈率": float(row["市盈率-动态"]) if row["市盈率-动态"] != "-" else None,
        }
    except Exception as e:
        st.error(f"获取数据失败：{e}")
        return None


@st.cache_data(ttl=300)  # 缓存5分钟
def get_kline_data(stock_code: str, days: int = 60):
    """
    获取个股历史K线数据
    参数:
        stock_code: 股票代码
        days: 获取最近多少天的数据
    返回:
        DataFrame: 包含日期、开高低收、成交量的数据表
    """
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # 前复权
        )

        return df
    except Exception as e:
        st.error(f"获取K线数据失败：{e}")
        return None


# ============================================================
# 第4部分：数据展示与可视化
# ============================================================

def show_price_card(data: dict):
    """展示股价指标卡片"""
    # 涨跌颜色：红涨绿跌（A股习惯）
    change_color = "#ff4d4f" if data["涨跌额"] >= 0 else "#52c41a"
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

    # 第二行指标
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


def show_kline_chart(kline_df):
    """展示K线图与成交额柱状图"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("K线图（前复权）", "成交额"),
    )

    # ---- K线图 ----
    fig.add_trace(
        go.Candlestick(
            x=kline_df["日期"],
            open=kline_df["开盘"],
            high=kline_df["最高"],
            low=kline_df["最低"],
            close=kline_df["收盘"],
            name="K线",
            increasing_line_color="#ff4d4f",   # 阳线红色
            decreasing_line_color="#52c41a",   # 阴线绿色
        ),
        row=1, col=1,
    )

    # ---- 成交额柱状图 ----
    colors = [
        "#ff4d4f" if close >= open_ else "#52c41a"
        for close, open_ in zip(kline_df["收盘"], kline_df["开盘"])
    ]

    fig.add_trace(
        go.Bar(
            x=kline_df["日期"],
            y=kline_df["成交额"],
            name="成交额",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2, col=1,
    )

    # 布局调整
    fig.update_layout(
        title_text=None,
        xaxis_rangeslider_visible=False,
        height=550,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Y轴标签
    fig.update_yaxes(title_text="价格（元）", row=1, col=1)
    fig.update_yaxes(title_text="成交额（元）", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 第5部分：主程序逻辑
# ============================================================

# 常用股票快速选择
with st.expander("📋 常用股票代码参考"):
    common_stocks = {
        "平安银行": "000001", "万科A": "000002", "中国平安": "601318",
        "贵州茅台": "600519", "比亚迪": "002594", "宁德时代": "300750",
        "招商银行": "600036", "五粮液": "000858", "中芯国际": "688981",
        "中兴通讯": "000063", "科大讯飞": "002230", "寒武纪": "688256",
    }
    cols = st.columns(6)
    for i, (name, code) in enumerate(common_stocks.items()):
        with cols[i % 6]:
            st.code(f"{code}  {name}")

# 查询逻辑
if query_btn or refresh_btn or "last_data" not in st.session_state:
    with st.spinner("⏳ 正在获取实时行情数据..."):
        data = get_realtime_price(stock_code)

    if data is None:
        st.warning(f"❌ 未找到股票代码 {stock_code}，请检查后重试")
    else:
        st.session_state["last_data"] = data
        st.success(f"✅ {data['名称']}（{stock_code}）行情加载成功 | 更新时间：{datetime.now().strftime('%H:%M:%S')}")

        # 显示指标卡片
        st.subheader(f"📊 {data['名称']} — 实时行情")
        show_price_card(data)

        # 显示K线图
        st.subheader("📈 K线走势（近60日）")
        kline_df = get_kline_data(stock_code, days=60)
        if kline_df is not None and not kline_df.empty:
            show_kline_chart(kline_df)
        else:
            st.info("暂无K线数据")

        # 原始数据表（折叠显示）
        with st.expander("📋 查看原始数据"):
            st.dataframe(kline_df.tail(20), use_container_width=True)

else:
    # 页面首次加载时的默认展示
    with st.spinner("⏳ 正在获取实时行情数据..."):
        data = get_realtime_price(stock_code)

    if data:
        st.session_state["last_data"] = data
        st.success(f"✅ {data['名称']}（{stock_code}）行情加载成功 | 更新时间：{datetime.now().strftime('%H:%M:%S')}")

        st.subheader(f"📊 {data['名称']} — 实时行情")
        show_price_card(data)

        st.subheader("📈 K线走势（近60日）")
        kline_df = get_kline_data(stock_code, days=60)
        if kline_df is not None and not kline_df.empty:
            show_kline_chart(kline_df)

# 页脚
st.divider()
st.caption("⚠️ 声明：本工具仅用于学习与技术展示，不构成任何投资建议。股市有风险，投资需谨慎。")

