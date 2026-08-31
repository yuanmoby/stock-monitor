# 📈 股票行情监测 + RAG + LSTM 项目合集

个人 AI 学习项目，覆盖 AI 应用开发的核心链路：**数据获取 → 数据处理 → 可视化 → 向量检索 → 模型训练**。

## 项目清单

| 项目 | 文件 | 技术栈 | 核心能力 |
|------|------|--------|---------|
| 实时股票行情监测 | `app.py` | Streamlit / Plotly / Pandas | API 对接、双层缓存、交互式可视化 |
| RAG 财报问答 Demo | `rag_demo.py` | sentence-transformers / ChromaDB | Embedding 向量化、向量检索、Prompt 工程 |
| LSTM 股价涨跌预测 | `lstm_predict.py` | PyTorch | 时间序列建模、模型训练与评估 |
| 数据源封装（共用） | `data_source.py` | requests / akshare | 接口风控应对、重试退避、本地缓存降级 |

## 环境准备

```bash
# 1. 安装 Python 3.10+（安装时勾选 Add Python to PATH）

# 2. 安装依赖（国内建议用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Windows 提示：如果 `import torch` 报 `WinError 1114 / c10.dll` 加载失败，
> 说明缺少新版 Visual C++ 运行库，到微软官网下载安装
> [最新版 VC++ 2015-2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 即可。

## 运行

```bash
# 项目1：股票行情监测（浏览器自动打开 localhost:8501）
streamlit run app.py

# 项目2：RAG Demo（首次运行自动下载约100MB中文Embedding模型）
python rag_demo.py
# 若模型下载慢，先执行：set HF_ENDPOINT=https://hf-mirror.com

# 项目3：LSTM 预测（约1-2分钟出结果）
python lstm_predict.py
```

## 上传 GitHub（简历里放链接，可信度提升十倍）

```bash
cd Desktop/stock-monitor
git init
git add .
git commit -m "股票行情监测 + RAG + LSTM 学习项目"
# 在 github.com 新建仓库 stock-monitor 后：
git remote add origin https://github.com/你的用户名/stock-monitor.git
git push -u origin main
```

## 面试要点：每个"为什么"都在这份代码里

### data_source.py — 数据层设计（本项目最有含金量的部分）

- **为什么实时报价走"单股接口"而不是全市场快照？** 全市场快照接口要分 59 页才能拉完 5000+ 只股票，连续请求极易触发东财风控断连（实测 curl 也会被断）。单股接口一次请求拿一只股票，既快又不触发风控。这是踩坑后改的设计。
- **为什么每个请求都带指数退避重试？** 东财接口的断连往往成串出现，间隔太短的重试没有意义。所以重试间隔逐次拉长（2.5s、5s、7.5s…），共 4 次。
- **为什么默认 UA 会被拒？** 东财风控拒绝默认的 `python-requests` 请求头（表现是 `RemoteDisconnected`）。通过覆写 `requests.utils.default_user_agent` 统一替换成浏览器 UA，对 akshare 内部的请求也生效。
- **为什么要本地缓存降级？** 接口彻底不可用时不至于演示翻车：自动用上次成功拉取的数据（存在 `data_cache/` 目录），并在页面上明确提示"当前为缓存数据"。演示要永远跑得起来。

### app.py 的设计决策

- **为什么报价缓存 30 秒、K 线缓存 5 分钟？** 报价是实时数据，30 秒平衡新鲜度和请求量；K 线按日更新，盘中最多只影响最后一根，5 分钟足够。
- **为什么 K 线取自然日区间时要 ×2？** 接口按自然日取区间，但股市只有交易日，约 90 个自然日才含 60 个交易日，所以取 2 倍再截尾部。
- **为什么用前复权（qfq）？** 分红除权会在 K 线上留下跳空缺口，前复权把历史价格按除权折算，走势才连贯。
- **输入框切代码就自动加载是怎么实现的？** 用 `st.session_state["shown_code"]` 记录当前展示的代码，与输入框值不一致就触发加载；按钮负责手动刷新。
- **MA5/MA10 均线为什么加？** 均线是观察短期趋势最基础的指标，用 `rolling(5).mean()` 两行就能算出来，却让图的信息量大增。

### rag_demo.py 的设计决策

- **为什么用 BGE 模型？** 智源开源的中文 Embedding 模型，中文语义效果好，免费可商用；财报术语的中文向量表示优于通用英文模型。
- **为什么向量库指定余弦空间？** BGE 训练目标基于余弦相似度（向量方向比长度重要），ChromaDB 默认是 L2 距离，所以显式指定 `hnsw:space=cosine`，相似度 = 1 - distance。
- **RAG 的本质一句话：** 把知识库向量化存起来，问题同样向量化去检索 Top-K 最相似的，把检索结果塞进 Prompt 让 LLM 基于真实资料回答——解决幻觉问题。

### lstm_predict.py 的设计决策

- **为什么准确率只有 50%-60%？** 这恰好证明股价预测不是简单模式匹配。政策、情绪、新闻才是更大的影响因子。如果准确率虚高（80%+），反而要怀疑数据泄露——对结果保持审慎是机器学习的基本素养。
- **为什么不用 Transformer？** 几百条日线数据用 LSTM 足够；Transformer 的自注意力需要更多数据才能学好长距离依赖，小数据集上 LSTM 的顺序处理归纳偏置是优势。
- **为什么按时间顺序切分数据集？** 未来数据不能用来训练，否则就是数据泄露；随机打乱会让模型"偷看"未来。
- **为什么要和"盲猜基线"比？** 涨跌二分类随机猜是 50%，多数类基线更高；模型必须超过它才算学到了东西。脚本会自动对比并如实报告，这是最基础的评估素养。

## 常见问题

| 问题 | 解决 |
|------|------|
| `pip install` 慢/失败 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `import torch` 报 c10.dll 错误 | 安装最新版 VC++ 运行库（见"环境准备"） |
| 行情接口偶发失败 | 重试机制 + 本地缓存自动降级，等几分钟即可恢复；接口对高频请求有风控，正常使用频率不会触发 |
| RAG 模型下载慢 | `set HF_ENDPOINT=https://hf-mirror.com` |
| 页面加载慢 | 首次拉取约 10-20 秒，之后走缓存 |

## 免责声明

本项目仅用于学习与技术展示，不构成任何投资建议。股市有风险，投资需谨慎。
