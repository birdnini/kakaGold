import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optimizer_lib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import warnings
import random

warnings.filterwarnings("ignore")

# ==========================================
# ⚡ 上帝模式：锁死全局宇宙随机种子
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# 1. 终极数据引擎：加装 RSI & MACD 动能引擎
# ==========================================
def load_and_clean_data(csv_file="kaka_macro_log.csv", seq_length=5):
    print("🚀 启动 [高频信号调理 & 动能雷达提取引擎] ...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='gbk')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None, None, 0

    df = df.iloc[:, -5:]
    df.columns = ['Time', 'Gold', 'Oil', 'DXY', 'NVDA']
    features_raw = ['Gold', 'Oil', 'DXY', 'NVDA']

    for col in features_raw:
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().bfill()

    raw_gold_array = df['Gold'].values.copy()

    # ---------------------------------------------------------
    # ⚡ 特征工程 (Feature Engineering) V2.0：重火力全开
    # ---------------------------------------------------------
    # 基础物理特征
    df['Gold_Ret'] = df['Gold'].pct_change()
    df['Oil_Ret'] = df['Oil'].pct_change()
    df['DXY_Ret'] = df['DXY'].pct_change()
    df['NVDA_Ret'] = df['NVDA'].pct_change()
    df['Gold_MA5'] = df['Gold'].rolling(window=5).mean()
    df['Gold_Bias'] = (df['Gold'] - df['Gold_MA5']) / df['Gold_MA5']
    df['Gold_Oil_Ratio'] = df['Gold'] / df['Oil']
    df['Gold_DXY_Ratio'] = df['Gold'] / df['DXY']

    # 🚀 新增引擎 A：RSI 弹簧过载传感器 (14周期)
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)  # 防止除以0引发核心熔毁
    df['Gold_RSI'] = 100 - (100 / (1 + rs))
    df['Gold_RSI'] = df['Gold_RSI'].fillna(100)  # 补全极值

    # 🚀 新增引擎 B：MACD 多普勒加速度计 (12,26,9)
    exp1 = df['Gold'].ewm(span=12, adjust=False).mean()
    exp2 = df['Gold'].ewm(span=26, adjust=False).mean()
    df['Gold_MACD'] = exp1 - exp2  # 我们直接把最核心的 DIF 线作为动能引脚喂给 AI！

    # 现在的输入引脚暴增到了 9 根！！！
    engineered_features = [
        'Gold_Ret', 'Oil_Ret', 'DXY_Ret', 'NVDA_Ret',
        'Gold_Bias', 'Gold_Oil_Ratio', 'Gold_DXY_Ratio',
        'Gold_RSI', 'Gold_MACD'  # <--- 新接入的两路高频信号
    ]
    input_pins = len(engineered_features)
    print(f"🔌 硬件级升级完成：芯片输入引脚已扩容至 [ {input_pins} 维 ]！")

    valid_indices = df[engineered_features].dropna().index
    df_clean = df.loc[valid_indices].reset_index(drop=True)
    gold_unscaled_clean = raw_gold_array[valid_indices]

    if len(df_clean) <= seq_length:
        print("❌ 特征提取后，有效时序切片不足！")
        return None, None, 0

    data_matrix = df_clean[engineered_features].values
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = feature_scaler.fit_transform(data_matrix)

    X, y = [], []
    for i in range(len(X_scaled) - seq_length):
        window_x = X_scaled[i: i + seq_length]
        current_price = gold_unscaled_clean[i + seq_length - 1]
        future_price = gold_unscaled_clean[i + seq_length]
        X.append(window_x)
        y.append(1 if future_price > current_price else 0)

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor, input_pins


# ==========================================
# 2. 硬件描述语言：Kaka-LSTM 高阶概率芯片
# ==========================================
class KakaMacro_LSTM_Prob(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(KakaMacro_LSTM_Prob, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# ==========================================
# 3. 核心总控台：火控系统与可视化
# ==========================================
def train_classification_model():
    seq_length = 5
    X, y, input_pins = load_and_clean_data(csv_file="kaka_macro_log.csv", seq_length=seq_length)
    if X is None: return

    # 芯片自适应 9 根引脚
    model = KakaMacro_LSTM_Prob(input_size=input_pins, hidden_size=64, num_layers=2)
    criterion = nn.BCELoss()
    # 既然维度变高了，稍微降低一点学习率，让它学得更仔细
    optimizer = optimizer_lib.AdamW(model.parameters(), lr=0.0004, weight_decay=1e-4)

    epochs = 400
    print("================ [重装芯片烧录中] ================")
    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_probabilities = model(X).numpy()
        actual_directions = y.numpy()

    # 🎯 终极火控逻辑
    latest_prob = float(predicted_probabilities[-1][0])
    b = 1.0

    if latest_prob > 0.5:
        p = latest_prob
        kelly_f = (p * b - (1.0 - p)) / b
        raw_action_sym = "[ BUY ]"
        raw_action_txt = "买入预期上涨"
        action_color = '#A6E3A1'
    else:
        p = 1.0 - latest_prob
        kelly_f = (p * b - (1.0 - p)) / b
        raw_action_sym = "[ SELL ]"
        raw_action_txt = "做空预期下跌"
        action_color = '#F38BA8'

    quarter_kelly_pct = (kelly_f / 4) * 100
    final_position_pct = min(quarter_kelly_pct, 5.0)

    if final_position_pct < 1.0:
        display_action_sym = "[ IDLE ]"
        display_action_txt = "觀望：信号弱"
        action_color = '#6C7086'
        pos_size_for_display = 0.0
    else:
        display_action_sym = raw_action_sym
        display_action_txt = raw_action_txt
        pos_size_for_display = final_position_pct

    predictions_binary = (predicted_probabilities > 0.5).astype(int)
    correct = (predictions_binary == actual_directions).sum()
    accuracy = correct / len(actual_directions)

    # ==========================================
    # 4. GUI 【数字仪表盘】
    # ==========================================
    plt.style.use('dark_background')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

    # --- A. 顶部面板：数字仪表盘 ---
    ax_dash = fig.add_subplot(gs[0])
    ax_dash.axis('off')

    ax_dash.text(0.5, 1.2, 'Kaka-Terminal [火控重装版] - 加入 RSI/MACD 双动能引擎',
                 fontsize=14, color='white', ha='center', fontweight='bold')

    ax_dash.annotate(display_action_sym, xy=(0.02, 0.45), xycoords='axes fraction',
                     fontsize=24, fontweight='bold', color='black',
                     bbox=dict(facecolor=action_color, alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5'))
    ax_dash.text(0.18, 0.45, display_action_txt, fontsize=12, color=action_color, weight='bold',
                 verticalalignment='center')
    ax_dash.text(0.45, 0.45, f"预测未来会【上涨】的概率: {latest_prob * 100:.1f}%", fontsize=12, color='white',
                 verticalalignment='center')
    ax_dash.annotate(f"安全建/空仓比例: {pos_size_for_display:.2f}%", xy=(0.75, 0.45), xycoords='axes fraction',
                     fontsize=18, fontweight='bold', color='white',
                     verticalalignment='center',
                     bbox=dict(facecolor='#1E1E2E', alpha=0.5, edgecolor='gray', boxstyle='square,pad=0.6'))

    # --- B. 底部面板：示波器主画面 ---
    ax_osc = fig.add_subplot(gs[1])

    ax_osc.scatter(range(len(actual_directions)), actual_directions, color='#F9E2AF', s=30, zorder=5,
                   label='真实涨跌 (1=涨, 0=跌)')
    ax_osc.plot(predicted_probabilities, color='#CBA6F7', linestyle='-', linewidth=2.5, alpha=0.9,
                label='AI 预测上涨概率波形')
    ax_osc.axhline(y=0.5, color='#89DCEB', linestyle='--', alpha=0.6, label='50% 抛硬币参考线')

    # 图例修复：完美安置在最右侧，留足空间
    ax_osc.legend(['真实涨跌 (1=涨, 0=跌)', 'AI 预测概率波形', '50% 强弱分水岭'],
                  loc='upper left', bbox_to_anchor=(1.02, 1), prop={'size': 9}, frameon=True)

    ax_osc.set_title(f'多维特征融合后 拟合平均胜率 (Average: {accuracy * 100:.1f}%)', fontsize=11, color='gray')
    ax_osc.set_ylabel('采样电平 (Probability [0.0 - 1.0])', color='gray')
    ax_osc.set_xlabel('Recent Samples (Recent Sequence Index)', color='gray')
    ax_osc.set_ylim(-0.1, 1.1)
    ax_osc.grid(True, linestyle='--', alpha=0.1)

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    print("📈 AI 重装仪表盘生成完毕！请查看显存弹窗。")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists("kaka_macro_log.csv"):
        print("❌ 致命错误：找不到你的日志文件！")
    else:
        train_classification_model()