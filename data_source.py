"""
数据源封装模块：行情数据获取 + 本地缓存降级。
被 app.py 与 lstm_predict.py 共用，避免重复代码。

作者：齐北
日期：2026.08

设计说明（面试可讲）：
1. 实时报价走东财"单股行情"接口（一次请求拿一只股票），而不是
   全市场快照接口——后者要分 59 页才能拉完 5000+ 只股票，连续
   请求极易触发东财的风控断连。
2. 东财接口对高频请求会成串断开连接（实测 curl 也会被断），所以
   所有请求都带指数退避重试；K线另加"本地缓存降级"：接口彻底
   不可用时用上次成功拉取的数据顶上，保证演示可用。
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ------------------------------------------------------------
# 请求兼容处理（踩坑记录）
# 东财风控会拒绝默认的 python-requests 请求头（表现：RemoteDisconnected
# 连接被直接断开）。把默认 UA 替换成浏览器 UA 后恢复正常。
# 这里通过覆写 requests 的 default_user_agent 实现，对 akshare 内部
# 所有新创建的会话都生效，无需改动 akshare 源码。
# ------------------------------------------------------------

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

requests.utils.default_user_agent = lambda *args, **kwargs: BROWSER_UA

import akshare as ak  # noqa: E402  （在 UA 补丁之后导入）

# 本地缓存目录（放仓库里，接口不可用时演示也能跑）
CACHE_DIR = Path(__file__).parent / "data_cache"

# 东财单股实时行情接口
QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
# 字段含义：f43最新价 f44最高 f45最低 f46今开 f47成交量 f48成交额
#          f57代码 f58名称 f60昨收 f168换手率 f169涨跌额 f170涨跌幅 f162市盈率(动)
QUOTE_FIELDS = "f43,f44,f45,f46,f47,f48,f57,f58,f60,f168,f169,f170,f162"


def with_retries(func, attempts: int = 3, backoff: float = 1.5):
    """指数退避重试装饰器：失败后等 backoff*1、backoff*2 秒再试。

    参数的取舍（初版是 4 次重试、间隔 2.5s 起，最长要等 15 秒）：
    东财断连往往成串出现，重试太少熬不过"断连风暴"；但页面查询是
    交互场景，用户等 15 秒体验太差。折中为 3 次重试、1.5s 起递增，
    最长约 4.5 秒就降级到本地缓存——体验和韧性之间取平衡，
    兜底靠缓存预热（见 scripts/warm_cache.py）。"""

    def wrapper(*args, **kwargs):
        last_exc = None
        for i in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # 网络类异常统一重试
                last_exc = e
                time.sleep(backoff * (i + 1))
        raise last_exc

    return wrapper


def _secid(stock_code: str) -> str:
    """6位代码 → 东财 secid（沪市 1.xxxxxx，深市 0.xxxxxx）"""
    if stock_code.startswith(("6", "9", "5")):
        return f"1.{stock_code}"
    return f"0.{stock_code}"


def _get(url: str, params: dict, timeout: int = 10) -> requests.Response:
    """带浏览器 UA 的 GET 请求"""
    with requests.Session() as s:
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r


# ------------------------------------------------------------
# 实时报价
# ------------------------------------------------------------

def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _save_json(obj: dict, name: str):
    """尽力保存到磁盘。云端（如 Streamlit Cloud）文件系统只读，
    写入失败直接忽略——云端本来就自带内存缓存，不影响功能。"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        _cache_path(name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _load_json(name: str):
    p = _cache_path(name)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


@with_retries
def _fetch_quote(stock_code: str) -> dict | None:
    """调东财单股行情接口，返回行情字典；找不到该股时返回 None"""
    params = {"invt": 2, "fltt": 2, "fields": QUOTE_FIELDS, "secid": _secid(stock_code)}
    r = _get(QUOTE_URL, params)
    data = r.json().get("data")
    if data is None:
        return None

    def num(v):
        return float(v) if v not in (None, "-", "") else None

    quote = {
        "名称": data.get("f58"),
        "代码": data.get("f57"),
        "最新价": num(data.get("f43")),
        "涨跌额": num(data.get("f169")),
        "涨跌幅": num(data.get("f170")),
        "今开": num(data.get("f46")),
        "最高": num(data.get("f44")),
        "最低": num(data.get("f45")),
        "昨收": num(data.get("f60")),
        "成交量": num(data.get("f47")),   # 单位：手
        "成交额": num(data.get("f48")),   # 单位：元
        "换手率": num(data.get("f168")),
        "市盈率": num(data.get("f162")),
    }
    return quote if quote["最新价"] is not None else None


def get_quote(stock_code: str) -> tuple[dict | None, bool]:
    """获取实时报价。返回 (行情字典, 是否来自缓存)。

    三种结果区分清楚（对用户诚实）：
    - 接口正常且有数据 → (行情, False)
    - 接口失败但本地有缓存 → (缓存行情, True)
    - 接口失败且无缓存 → 抛异常（由调用方提示"接口暂不可用"）
    - 接口正常但查无此股 → (None, False)（由调用方提示"未找到"）
    """
    network_error = False
    try:
        quote = _fetch_quote(stock_code)  # None 表示确实没有这只股票
    except Exception:
        quote = None
        network_error = True
    if quote is not None:
        _save_json(quote, f"quote_{stock_code}.json")
        return quote, False
    cached = _load_json(f"quote_{stock_code}.json")
    if cached is not None:
        return cached, True
    if network_error:
        raise RuntimeError(f"行情接口暂不可用，且 {stock_code} 无本地缓存")
    return None, False


# ------------------------------------------------------------
# 历史K线
# ------------------------------------------------------------

@with_retries
def _fetch_kline(stock_code: str, days: int) -> pd.DataFrame:
    """调东财历史K线接口（akshare 封装），返回最近 days 个交易日。

    两个要点：
    1. 前复权（qfq）：把历史价格按分红除权折算，K 线走势才连贯。
    2. 接口按自然日取区间，但股市只有交易日。要凑满 days 个交易日，
       自然日区间要放大到约 2 倍，再取尾部截断。
    """
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    return df.tail(days)


def load_kline(stock_code: str, days: int = 60) -> tuple[pd.DataFrame, bool]:
    """获取历史K线。返回 (DataFrame, 是否来自缓存)。
    接口失败时降级到本地缓存，彻底不可用才抛异常。"""
    try:
        df = _fetch_kline(stock_code, days)
    except Exception:
        df = None
    if df is not None and not df.empty:
        # 尽力保存到磁盘；云端只读文件系统会写入失败，直接忽略
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(_cache_path(f"kline_{stock_code}_{days}.csv"), index=False, encoding="utf-8")
        except OSError:
            pass
        return df, False
    # 缓存文件名带天数，60日与330日的缓存互不串用
    p = _cache_path(f"kline_{stock_code}_{days}.csv")
    if p.exists():
        cached = pd.read_csv(p, encoding="utf-8")
        if not cached.empty:
            return cached, True
    raise RuntimeError(f"获取 {stock_code} K线失败，且无本地缓存可用")
