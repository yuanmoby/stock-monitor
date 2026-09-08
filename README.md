# 📈 股票行情监测 + RAG + LSTM + Agent 项目合集

个人 AI 学习项目，覆盖 AI 应用开发的核心链路：**数据获取 → 数据处理 → 可视化 → 向量检索 → 模型训练 → 智能体（Agent）**。

## 项目清单

| 项目 | 文件 | 技术栈 | 核心能力 |
|------|------|--------|---------|
| 实时股票行情监测 | `app.py` | Streamlit / Plotly / Pandas | API 对接、双层缓存、交互式可视化 |
| AI 股票问答 Agent | `agent.py` / `app_agent.py` | DeepSeek Function Calling / requests | 工具调用循环、大模型智能体 |
| RAG 财报问答 Demo | `rag_demo.py` | sentence-transformers / ChromaDB | Embedding 向量化、向量检索、Prompt 工程 |
| LSTM 股价涨跌预测 | `lstm_predict.py` | PyTorch | 时间序列建模、模型训练与评估 |
| 数据源封装（共用） | `data_source.py` | requests / akshare | 接口风控应对、重试退避、本地缓存降级 |

## 环境准备

```bash
# 1. 安装 Python 3.10+（安装时勾选 Add Python to PATH）

# 2. 网站（app.py）依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 两个 ML 脚本的依赖（只在本机学习用，不参与网站部署）
pip install -r requirements-ml.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> Windows 提示：如果 `import torch` 报 `WinError 1114 / c10.dll` 加载失败，
> 说明缺少新版 Visual C++ 运行库，到微软官网下载安装
> [最新版 VC++ 2015-2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 即可。

## 运行

```bash
# 项目1：股票行情监测（浏览器自动打开 localhost:8501）
streamlit run app.py
# 同一 WiFi 下的手机可访问 http://你的电脑IP:8501

# 项目2：AI 股票问答 Agent（网页聊天：streamlit run app_agent.py）
# 命令行版：python agent.py "茅台最近30天走势怎么样？"
# 需要 DeepSeek API Key：在项目目录新建 .env 文件写入 DEEPSEEK_API_KEY=sk-xxxx
# （到 https://platform.deepseek.com 注册创建，费用极低）

# 项目3：RAG Demo（首次运行自动下载约100MB中文Embedding模型）
# 代码已内置国内镜像（hf-mirror.com），直连超时自动走镜像，无需手动配置
python rag_demo.py

# 项目4：LSTM 预测（约1-2分钟出结果）
python lstm_predict.py

# 可选：给常用12只股票预热缓存（接口断连时演示不断档）
python scripts/warm_cache.py
```

## 部署到公网（免费，手机流量也能访问）

Streamlit Community Cloud 免费托管，任何设备任何网络都能打开：

1. 把本项目推送到 GitHub（见下一节）
2. 浏览器打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号登录
3. 点 "Create app" → 选仓库 `stock-monitor`、分支 `main`、入口文件 `app.py` → Deploy
4. 几分钟后得到公网地址：`https://你的账号-stock-monitor.streamlit.app`，手机 4G/5G 直接访问

> 免费版特点：网站闲置几分钟后会休眠，下次访问自动唤醒（约 1 分钟）；
> 云服务器在海外，行情接口偶发连不上时会自动显示仓库里预置的缓存数据。

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
- **为什么每个请求都带指数退避重试？** 东财接口的断连往往成串出现，间隔太短的重试没有意义，所以重试间隔逐次拉长。但页面查询是交互场景，不能等太久——初版 4 次重试最长等 15 秒，用户反馈太慢，折中为 **3 次重试、1.5s 起递增（1.5s、3s），最长约 4.5 秒就降级**到缓存，体验与韧性之间取平衡。
- **为什么有缓存预热脚本？** 查询一只没有缓存记录的股票，重试耗尽只能报错。`scripts/warm_cache.py` 会提前把常用 12 只股票的报价和 K 线缓存好（请求间隔 3 秒防风控），接口断连时这些股票依然能展示旧数据并带黄色提示，演示不断档。
- **为什么默认 UA 会被拒？** 东财风控拒绝默认的 `python-requests` 请求头（表现是 `RemoteDisconnected`）。通过覆写 `requests.utils.default_user_agent` 统一替换成浏览器 UA，对 akshare 内部的请求也生效。
- **为什么要本地缓存降级？** 接口彻底不可用时不至于演示翻车：自动用上次成功拉取的数据（存在 `data_cache/` 目录），并在页面上明确提示"当前为缓存数据"。演示要永远跑得起来。

### app.py 的设计决策

- **为什么报价缓存 30 秒、K 线缓存 5 分钟？** 报价是实时数据，30 秒平衡新鲜度和请求量；K 线按日更新，盘中最多只影响最后一根，5 分钟足够。
- **为什么 K 线取自然日区间时要 ×2？** 接口按自然日取区间，但股市只有交易日，约 90 个自然日才含 60 个交易日，所以取 2 倍再截尾部。
- **为什么用前复权（qfq）？** 分红除权会在 K 线上留下跳空缺口，前复权把历史价格按除权折算，走势才连贯。
- **输入框切代码就自动加载是怎么实现的？** 用 `st.session_state["shown_code"]` 记录当前展示的代码，与输入框值不一致就触发加载；按钮负责手动刷新。
- **MA5/MA10 均线为什么加？** 均线是观察短期趋势最基础的指标，用 `rolling(5).mean()` 两行就能算出来，却让图的信息量大增。

### agent.py — AI Agent 的设计决策（投 Agent 岗必讲）

- **Agent 和普通大模型对话的区别？** 普通对话模型只能凭训练记忆回答，查不了实时数据。Agent 给模型一份"工具清单"（JSON Schema 描述的函数），模型判断"这个问题需要查数据"，返回工具调用请求，程序执行后把结果回传，模型再生成回答——"提问 → 调用工具 → 拿结果 → 再回答"循环，最多 5 轮。
- **为什么手写循环、不用 LangChain？** 函数调用的底层就是 OpenAI 兼容协议：POST /chat/completions，body 里带 messages + tools；模型返回 tool_calls；执行后把结果以 role=tool 追加回历史再请求。手写一遍 100 行，机制完全透明，之后再学框架就知道每层在做什么。本项目用 requests 直连 DeepSeek 接口，连 SDK 都没用。
- **为什么工具失败返回错误文本而不是崩溃？** 把错误信息作为工具结果回传给模型，模型会自动换思路（比如换个关键词重查）。容错设计是 Agent 工程化的基本要求。
- **为什么温度设 0.3？** 行情问答要稳定准确，不需要创意；低温度减少乱调用工具的幻觉。
- **Agent 和 RAG 什么关系？** 互补：RAG 给模型"资料"，Agent 给模型"手脚"。RAG 解决"知识幻觉"，Agent 解决"能力幻觉"。生产系统常组合使用。
- **工具执行为什么复用 data_source？** Agent 的工具和股票网站共用同一套数据层（重试、缓存降级全继承），既避免重复代码，也保证 Agent 查到的数据和网站一致。
- **模型查不到股票就胡说怎么办？** 真实踩坑：问"宇树科技怎么样"，静态表里没有，模型凭旧训练知识断言"还没上市"——实际已上市（688836）。修复：① 代码速查改成东财实时搜索接口（三层兜底：接口→搜索缓存→本地表），覆盖全A股；② 系统提示词加硬规则"严禁用训练知识断言未上市，查不到只能说工具查不到"。Agent 系统的事实必须来自工具，不来自模型记忆。

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
| RAG 模型下载慢/超时 | 代码已自动走 hf-mirror.com 镜像；若仍失败，手动执行 `set HF_ENDPOINT=https://hf-mirror.com` 再运行 |
| 页面加载慢 | 首次拉取约 10-20 秒，之后走缓存 |

## 免责声明

本项目仅用于学习与技术展示，不构成任何投资建议。股市有风险，投资需谨慎。
