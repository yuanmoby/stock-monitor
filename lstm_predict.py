"""
LSTM 股价涨跌预测 — 用 PyTorch 实现的最简 LSTM 分类模型
作者：齐北
日期：2025.06 初版 / 2026.08 修正

运行前安装依赖：
    pip install torch
"""

import sys
import warnings

# Windows 中文控制台默认 GBK 编码，打印 emoji 会报 UnicodeEncodeError，
# 统一把标准输出切到 UTF-8（Win11 终端默认支持 UTF-8 显示）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

# 数据获取复用股票监测项目封装的 data_source 模块
from data_source import load_kline

print("=" * 60)
print("  LSTM 股价涨跌预测")
print("=" * 60)

# ============================================================
# 第一步：获取数据（复用股票监测项目的 akshare）
# ============================================================
print("\n📊 正在获取股票历史数据...")

# 拿贵州茅台最近约330个交易日（≈500自然日）的日线数据（前复权）
try:
    df, from_cache = load_kline("600519", days=330)
except RuntimeError as e:
    # 接口彻底不可用且无本地缓存：给出友好提示而不是堆错误栈
    print(f"❌ {e}")
    print("   建议：稍等几分钟（东财风控会自动解除）后重试，")
    print("   或先运行 `py scripts/warm_cache.py` 预热缓存。")
    raise SystemExit(1)
if from_cache:
    print("   ⚠️ 行情接口暂不可用，本次使用本地缓存数据（上次成功拉取）")
print(f"   获取到 {len(df)} 条日线数据")
print(f"   日期范围：{df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")

# ============================================================
# 第二步：构建特征和标签
# ============================================================
# 用收盘价做预测，同时加入成交额作为辅助特征
prices = df["收盘"].values.astype(np.float32)
volumes = df["成交额"].values.astype(np.float32)

# 标签：明天涨(1)还是跌(0)？明天的收盘价 > 今天的收盘价 → 涨
labels = np.where(prices[1:] > prices[:-1], 1, 0)

# 特征：用过去60天的 [收盘价, 成交额] 预测明天的涨跌
LOOKBACK = 60

X_list, y_list = [], []
for i in range(LOOKBACK, len(prices) - 1):
    price_window = prices[i - LOOKBACK: i]
    volume_window = volumes[i - LOOKBACK: i]
    # 归一化：除以窗口内的最大值，防止数值过大影响训练
    price_norm = price_window / (price_window.max() + 1e-8)
    volume_norm = volume_window / (volume_window.max() + 1e-8)
    features = np.stack([price_norm, volume_norm], axis=1)

    X_list.append(features)
    y_list.append(labels[i])

X = np.array(X_list)  # 形状：(样本数, 60天, 2个指标)
y = np.array(y_list)  # 形状：(样本数,)

print(f"\n📐 数据形状：")
print(f"   X (特征)：{X.shape}")
print(f"   y (标签)：{y.shape}")
print(f"   涨跌比例：涨 {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

# ============================================================
# 第三步：划分训练集和测试集（80%训练，20%测试）
# ============================================================
# 按时间顺序切分，不能随机打乱——未来数据不能用来训练
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"\n   训练集：{len(X_train)} 样本")
print(f"   测试集：{len(X_test)} 样本")

# ============================================================
# 第四步：构建 LSTM 模型（核心）
# ============================================================
class StockLSTM(nn.Module):
    """最简 LSTM 分类器：2层LSTM → Dropout → 全连接 → 2分类"""
    def __init__(self, input_size=2, hidden_size=32, num_layers=2):
        super().__init__()
        # LSTM 层：输入 2 个特征（价格+成交额），隐藏层 32 维，堆叠 2 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        # Dropout：训练时随机丢弃 30% 神经元，防止过拟合
        self.dropout = nn.Dropout(0.3)
        # 全连接层：把 LSTM 输出映射到 2 个类别（涨/跌）
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]   # 只取最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)
        return out


model = StockLSTM(input_size=2, hidden_size=32, num_layers=2)
criterion = nn.CrossEntropyLoss()          # 二分类用交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 LSTM 模型参数量：{total_params:,}")
print(f"   结构：LSTM(2→32, 2层) → Dropout → Linear(32→2)")

# ============================================================
# 第五步：训练模型
# ============================================================
print(f"\n🚀 开始训练...")
BATCH_SIZE = 32
EPOCHS = 50

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train_t), BATCH_SIZE):
        batch_X = X_train_t[i: i + BATCH_SIZE]
        batch_y = y_train_t[i: i + BATCH_SIZE]

        outputs = model(batch_X)            # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        optimizer.zero_grad()
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / (len(X_train_t) // BATCH_SIZE + 1)
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ============================================================
# 第六步：评估模型
# ============================================================
print(f"\n📊 模型评估：")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_t).float().mean().item()

print(f"   测试集准确率：{accuracy*100:.2f}%")

# 盲猜基线：如果一直猜"涨"（多数类），能拿多少分？
# 模型必须超过这个基线，才说明真的学到了东西
baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"   盲猜'总是涨'基线：{baseline*100:.2f}%")
if accuracy > baseline:
    print(f"   ✅ 模型比盲猜强 {accuracy - baseline:.2%}！")
else:
    print(f"   ⚠️ 模型还没超过盲猜，可能原因：数据太少/特征太简单/需要调参")

# ============================================================
# 第七步：预测下一个交易日
# ============================================================
print(f"\n🔮 预测下一个交易日：")
last_window_price = prices[-LOOKBACK:] / (prices[-LOOKBACK:].max() + 1e-8)
last_window_volume = volumes[-LOOKBACK:] / (volumes[-LOOKBACK:].max() + 1e-8)
last_features = np.stack([last_window_price, last_window_volume], axis=1)
last_tensor = torch.FloatTensor(last_features).unsqueeze(0)  # 加 batch 维度

with torch.no_grad():
    pred_output = model(last_tensor)
    pred_prob = torch.softmax(pred_output, dim=1)[0]
    pred_class = torch.argmax(pred_output, dim=1).item()

print(f"   预测结果：{'📈 涨' if pred_class == 1 else '📉 跌'}")
print(f"   涨的概率：{pred_prob[1]*100:.1f}%")
print(f"   跌的概率：{pred_prob[0]*100:.1f}%")

print(f"\n{'─' * 50}")
print(f"⚠️ 声明：此预测仅供学习演示，不构成任何投资建议！")
print(f"   股价受政策、情绪、宏观等数百因素影响，LSTM仅捕捉了价格趋势模式。")
print(f"{'─' * 50}")
