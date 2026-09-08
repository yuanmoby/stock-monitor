# -*- coding: utf-8 -*-
"""
缓存预热脚本：给常用股票提前拉好报价和K线，写入 data_cache/ 目录。

为什么需要：东财接口断连成串出现，查询一只没有缓存的股票时，
重试耗尽就只能报错。提前把常用 12 只股票的数据缓存好，
接口断连时页面依然能展示（旧数据 + 黄色提示），演示不断档。

用法：
    python scripts/warm_cache.py
注意：请求之间留了 3 秒间隔，避免连续请求触发东财风控。
"""

import sys
import time
from pathlib import Path

# 让脚本能被直接运行（import data_source）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_source import get_quote, load_kline  # noqa: E402

COMMON_STOCKS = {
    "平安银行": "000001", "万科A": "000002", "中国平安": "601318",
    "贵州茅台": "600519", "比亚迪": "002594", "宁德时代": "300750",
    "招商银行": "600036", "五粮液": "000858", "中芯国际": "688981",
    "中兴通讯": "000063", "科大讯飞": "002230", "寒武纪": "688256",
}

INTERVAL = 5  # 请求间隔（秒），太快会触发风控；失败重跑时建议 8 秒以上


def main():
    ok_quote, ok_kline, fail = 0, 0, 0
    for name, code in COMMON_STOCKS.items():
        try:
            quote, from_cache = get_quote(code)
            time.sleep(INTERVAL)
            kline, from_cache_k = load_kline(code, days=60)
            time.sleep(INTERVAL)
            price = quote["最新价"] if quote else "?"
            q_tag = "缓存" if from_cache else "实时"
            k_tag = "缓存" if from_cache_k else "实时"
            ok_quote += 1
            ok_kline += 1
            print(f"[{name} {code}] 报价{price}（{q_tag}）| K线{len(kline)}根（{k_tag}）")
        except Exception as e:
            fail += 1
            print(f"[{name} {code}] 失败：{str(e)[:60]}（重试耗尽，跳过，下次再预热）")
            time.sleep(INTERVAL)
    # LSTM 脚本固定用茅台的330个交易日数据，单独预热一份长窗口缓存
    try:
        df_lstm, fc_lstm = load_kline("600519", days=330)
        time.sleep(INTERVAL)
        print(f"[贵州茅台 600519] LSTM用330日K线：{len(df_lstm)}根（{'缓存' if fc_lstm else '实时'}）")
    except Exception as e:
        print(f"[贵州茅台 600519] LSTM用330日K线失败：{str(e)[:50]}（可稍后单独重试）")

    print(f"\n完成：报价 {ok_quote}/12，K线 {ok_kline}/12，失败 {fail}")


if __name__ == "__main__":
    main()
