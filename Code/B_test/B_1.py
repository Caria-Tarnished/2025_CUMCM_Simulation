import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Matplotlib 中文显示设置
# 请确保您的系统中安装了支持中文的字体，例如 'SimHei'
try:
    font = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=12)
except IOError:
    # 如果找不到 SimHei，使用默认字体 (可能会显示方框)
    font = FontProperties(size=12)
    print("警告：未找到 'SimHei' 字体，图表中的中文可能无法正常显示。")
    print("请将代码中的字体路径 'c:\\windows\\fonts\\simhei.ttf' 替换为您系统中的有效中文字体路径。")


# --- 1. 定义模型参数 ---
# 这些参数直接来自问题描述和您的 .md 分析文件

p0 = 0.10  # 可接收质量水平 (AQL), H0: p=p0
p1 = 0.20  # 可拒绝质量水平 (RQL), H1: p=p1
alpha = 0.10 # 生产者风险 (第一类错误概率)
beta = 0.05  # 消费者风险 (第二类错误概率)

print("--- 模型输入参数 ---")
print(f"可接收质量水平 (AQL) p0: {p0:.2f}")
print(f"可拒绝质量水平 (RQL) p1: {p1:.2f}")
print(f"生产者风险 alpha (α): {alpha:.2f}")
print(f"消费者风险 beta (β): {beta:.2f}\n")


# --- 2. 计算 SPRT 决策边界参数 ---
# 根据 Wald 的序贯分析理论计算斜率和截距

# 计算似然比的两个边界 A 和 B
A = (1 - beta) / alpha
B = beta / (1 - alpha)

# 为了方便计算，取对数
log_A = math.log(A)
log_B = math.log(B)

# 计算对数似然比中的常用项
g0 = math.log(p1 / p0)
g1 = math.log((1 - p0) / (1 - p1))

# 计算决策线的斜率 (s) 和截距 (h0, h1)
s = g1 / (g0 + g1)
h0 = -log_B / (g0 + g1)
h1 = log_A / (g0 + g1)

print("--- SPRT 决策规则计算结果 ---")
print(f"决策线斜率 (s): {s:.4f}")
print(f"接收线截距 (-h0): {-h0:.4f}")
print(f"拒收线截距 (h1): {h1:.4f}\n")

print("--- 最终抽样检测方案 ---")
print("在累计抽取 n 个样本后，发现其中有 d_n 个次品:")
print(f"1. 如果 d_n >= {s:.4f}n + {h1:.4f}, 则【拒收】该批次。")
print(f"2. 如果 d_n <= {s:.4f}n - {h0:.4f}, 则【接收】该批次。")
print(f"3. 否则，继续抽样。\n")


# --- 3. 可视化决策边界 ---
def plot_sprt_boundaries(s, h0, h1, max_n=150):
    """
    绘制 SPRT 决策边界图。

    参数:
    s (float): 决策线斜率
    h0 (float): 接收线截距参数
    h1 (float): 拒收线截距参数
    max_n (int): 图表中显示的最大样本数量
    """
    n_values = np.arange(1, max_n + 1)
    
    # 计算接收线和拒收线的y值
    # d_n >= sn + h1  => 拒收
    # d_n <= sn - h0  => 接收
    rejection_line = s * n_values + h1
    acceptance_line = s * n_values - h0
    
    # 确保接收线的值不为负，因为次品数不能是负数
    acceptance_line[acceptance_line < 0] = 0

    plt.figure(figsize=(12, 8))
    
    # 绘制决策线
    plt.plot(n_values, rejection_line, label='拒收线 (d = sn + h?)', color='red', linestyle='--')
    plt.plot(n_values, acceptance_line, label='接收线 (d = sn - h?)', color='green', linestyle='--')
    
    # 填充不同决策区域
    plt.fill_between(n_values, rejection_line, max(rejection_line)*1.1, color='red', alpha=0.1, label='拒收区域')
    plt.fill_between(n_values, acceptance_line, rejection_line, color='yellow', alpha=0.2, label='继续抽样区域')
    plt.fill_between(n_values, 0, acceptance_line, color='green', alpha=0.1, label='接收区域')

    # 设置图表标题和标签
    plt.title('SPRT 抽样决策边界', fontproperties=font, fontsize=16)
    plt.xlabel('样本数量 (n)', fontproperties=font, fontsize=14)
    plt.ylabel('累计次品数量 (d?)', fontproperties=font, fontsize=14)
    
    # 设置坐标轴范围和刻度
    plt.xlim(0, max_n)
    plt.ylim(0, max(rejection_line) * 0.8) # 调整y轴范围以便更好地观察
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 显示图例
    # 获取图例句柄和标签，并设置字体
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, prop=font)

    # 寻找并标注关键决策点
    # 首次可能接收的点 (d=0)
    n_accept_zero_defect = math.ceil(h0 / s)
    plt.scatter([n_accept_zero_defect], [0], color='blue', s=100, zorder=5)
    plt.text(n_accept_zero_defect + 2, 0.5, f'在第 {n_accept_zero_defect} 个样本时\n若无次品即可接收', 
             fontproperties=font, color='blue')
    
    print(f"--- 关键决策点 ---")
    print(f"如果连续抽样未发现次品，最早可在抽取第 {n_accept_zero_defect} 个样本后做出【接收】决策。")

    plt.show()

# 调用函数生成图表
plot_sprt_boundaries(s, h0, h1)
