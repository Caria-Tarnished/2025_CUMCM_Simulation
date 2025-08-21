import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
import pandas as pd
import os

# --- 1. 定义常量与参数 (SI units) ---
# 板凳数量
NUM_BENCHES = 223
# 把手/节点数量
NUM_HANDLES = NUM_BENCHES + 1

# 几何尺寸
L_HEAD = 3.41  # 龙头板长 (m)
L_BODY = 2.20  # 龙身板长 (m)
D_HANDLE = 0.275 # 把手中心到板头的距离 (m)

# 计算有效的连杆长度 (把手中心距)
L_LINK_HEAD = L_HEAD - 2 * D_HANDLE  # 龙头 (H1-H2)
L_LINK_BODY = L_BODY - 2 * D_HANDLE  # 龙身 (Hk-H(k+1))

# 创建连杆长度数组 (索引0对应H1-H2, 索引1对应H2-H3, ...)
LINK_LENGTHS = np.array([L_LINK_HEAD] + [L_LINK_BODY] * (NUM_BENCHES - 1))

# 运动参数
V_HEAD = 1.0  # 龙头前把手速度 (m/s)
PITCH = 0.55  # 螺距 (m)
B_FACTOR = PITCH / (2 * np.pi) # 螺线方程 r = b * theta

# 初始条件参数
START_LOOP = 16
THETA_START = START_LOOP * 2 * np.pi

# 仿真时间
T_START, T_END = 0, 300
T_EVAL = np.arange(T_START, T_END + 1, 1)

# --- 2. 定义螺线路径函数 ---

def get_r(theta):
    """根据极角计算螺线半径"""
    return B_FACTOR * theta

def get_cartesian(theta):
    """根据极角计算笛卡尔坐标 (顺时针盘入)"""
    r = get_r(theta)
    x = r * np.cos(theta)
    y = -r * np.sin(theta)  # 负号用于实现顺时针
    return np.array([x, y])

def arc_length_integrand(theta):
    """弧长积分的被积函数 ds/dtheta"""
    # 避免在theta=0时出现不稳定的情况
    if np.abs(theta) < 1e-9:
        return B_FACTOR
    return np.sqrt(get_r(theta)**2 + B_FACTOR**2)

def get_arc_length(theta_start, theta_end):
    """计算从 theta_start 到 theta_end 的弧长 (积分方向为角度增大的方向)"""
    length, _ = quad(arc_length_integrand, theta_start, theta_end)
    return length

def get_tangent_vector(theta):
    """计算单位切向量"""
    dx_dtheta = B_FACTOR * (np.cos(theta) - theta * np.sin(theta))
    dy_dtheta = -B_FACTOR * (np.sin(theta) + theta * np.cos(theta))
    vec = np.array([dx_dtheta, dy_dtheta])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.array([1.0, 0.0]) # 在原点附近，近似为沿x轴
    return vec / norm

# --- 3. 计算初始条件 (t=0) ---
print("正在计算初始条件 (t=0)，这可能需要一些时间...")
initial_positions = np.zeros((NUM_HANDLES, 2))

# H1 (龙头前把手) 的初始位置和角度
theta_h1_initial = THETA_START
initial_positions[0, :] = get_cartesian(theta_h1_initial)

# 迭代计算 H2 到 H224 的初始位置
# 我们需要从H1沿着螺线向后找点
current_theta = theta_h1_initial
for i in range(NUM_BENCHES): # i from 0 to 222
    # 目标：找到一个theta_prev，使得从theta_prev到current_theta的弧长等于连杆长度
    target_length = LINK_LENGTHS[i]
    
    # 定义求解函数: 弧长(theta_prev, current_theta) - 目标长度 = 0
    def find_root_func(theta_prev_scalar):
        return get_arc_length(theta_prev_scalar, current_theta) - target_length

    # 使用 fsolve 寻找下一个点的角度
    # 初始猜测值比当前角度小一点，因为是向后找
    solution = fsolve(find_root_func, current_theta - target_length / get_r(current_theta))
    prev_theta = solution[0]
    
    initial_positions[i + 1, :] = get_cartesian(prev_theta)
    current_theta = prev_theta # 更新当前点，为下一次迭代做准备

print("初始条件计算完成。")


# --- 4. 构建ODE系统 ---
def ode_system(t, y):
    """
    定义常微分方程组
    y: 状态向量 [x2, y2, x3, y3, ..., x224, y224, theta1]
    """
    # 初始化导数向量 (速度)
    dydt = np.zeros_like(y)
    
    # 从状态向量中提取龙头H1的当前角度
    theta1 = y[-1]
    
    # 根据角度计算龙头H1的当前位置和速度
    pos_h1 = get_cartesian(theta1)
    tangent_h1 = get_tangent_vector(theta1)
    vel_h1 = V_HEAD * tangent_h1
    
    # 构造包含所有把手位置的完整数组，以便于迭代
    # H1的位置是计算得出的，H2及以后是ODE的状态变量
    all_pos = np.zeros((NUM_HANDLES, 2))
    all_pos[0, :] = pos_h1
    all_pos[1:, :] = y[:-1].reshape((NUM_BENCHES, 2))

    # 核心递推逻辑：计算H2到H224的速度
    current_vel = vel_h1
    for i in range(NUM_BENCHES): # i from 0 to 222
        # H_{i+1} 的速度由 H_{i} 决定
        pos_front = all_pos[i]
        pos_back = all_pos[i+1]
        
        link_vec = pos_front - pos_back
        dist = np.linalg.norm(link_vec)
        
        # 避免除零错误
        if dist < 1e-9:
            unit_vec = np.zeros(2)
        else:
            unit_vec = link_vec / dist
            
        # 核心公式：后点速度 = 前点速度在连线方向上的投影
        scalar_vel = np.dot(current_vel, unit_vec)
        next_vel = scalar_vel * unit_vec

        # 将计算出的H_{i+2}的速度存入导数向量
        # 状态向量y中，H2的速度是dydt[0:2], H3是dydt[2:4], ...
        dydt[2*i : 2*i+2] = next_vel
        
        # 为下一次迭代更新 "当前速度"
        current_vel = next_vel

    # 计算龙头角度的变化率 d(theta)/dt = v / (ds/dtheta)
    # 因为是顺时针向内盘入，角度减小，所以速率为负
    dtheta_dt = -V_HEAD / arc_length_integrand(theta1)
    dydt[-1] = dtheta_dt
    
    return dydt

# --- 5. 运行仿真 ---
print("开始进行ODE仿真 (0s to 300s)...")
# 初始状态向量 y0 = [x2(0), y2(0), ..., x224(0), y224(0), theta1(0)]
y0 = np.concatenate([initial_positions[1:].flatten(), [theta_h1_initial]])

# 使用 solve_ivp 求解
solution = solve_ivp(
    ode_system,
    [T_START, T_END],
    y0,
    method='RK45', # 'RK45' 是一个精度和效率很好的自适应方法
    t_eval=T_EVAL,
    dense_output=True # 允许在任意时间点插值
)

print("仿真完成。")

# --- 6. 后处理与结果输出 ---
print("正在处理结果并生成文件...")
# 提取所有时间点的解
sol_y = solution.y

# 提取 H2-H224 的位置
positions_bodies = sol_y[:-1, :].T.reshape(-1, NUM_BENCHES, 2)

# 提取并计算 H1 的位置
thetas_h1 = sol_y[-1, :]
positions_h1 = np.array([get_cartesian(th) for th in thetas_h1])

# 合并成完整的位置矩阵 (Time x Handle x Coords)
all_positions = np.zeros((len(T_EVAL), NUM_HANDLES, 2))
all_positions[:, 0, :] = positions_h1
all_positions[:, 1:, :] = positions_bodies

# 计算所有时间点的速度
all_velocities = np.zeros_like(all_positions)
for i, t in enumerate(T_EVAL):
    y_t = solution.sol(t) # 使用 dense_output 获取该时刻的精确状态
    derivatives = ode_system(t, y_t)
    
    # 计算 H1 的速度
    theta1_t = y_t[-1]
    tangent_h1_t = get_tangent_vector(theta1_t)
    all_velocities[i, 0, :] = V_HEAD * tangent_h1_t
    
    # 存入 H2-H224 的速度
    all_velocities[i, 1:, :] = derivatives[:-1].reshape(NUM_BENCHES, 2)

# 计算速度大小 (速率)
all_speeds = np.linalg.norm(all_velocities, axis=2)

# ---- 生成 result1.xlsx ----
# 创建两个DataFrame，一个用于位置，一个用于速度
pos_columns = ['Time']
vel_columns = ['Time']
labels = ['龙头'] + [f'第{i}节龙身' for i in range(1, NUM_BENCHES)] + ['龙尾'] + ['龙尾(后)']

for i in range(NUM_HANDLES):
    label = labels[i]
    if i == NUM_BENCHES: # 跳过"龙尾"前把手，因为它和"第222节龙身"前把手是同一个
        continue
    pos_columns.extend([f'{label}_x', f'{label}_y'])
    vel_columns.append(f'{label}_v')

pos_data = []
vel_data = []

for t_idx, t in enumerate(T_EVAL):
    pos_row = [t]
    vel_row = [t]
    for h_idx in range(NUM_HANDLES):
        if h_idx == NUM_BENCHES:
            continue
        pos_row.extend(all_positions[t_idx, h_idx, :])
        vel_row.append(all_speeds[t_idx, h_idx])
    pos_data.append(pos_row)
    vel_data.append(vel_row)

df_pos = pd.DataFrame(pos_data, columns=pos_columns)
df_vel = pd.DataFrame(vel_data, columns=vel_columns)

# 使用ExcelWriter将两个DataFrame写入同一个文件的不同sheet
output_filename = "result1.xlsx"
with pd.ExcelWriter(output_filename) as writer:
    df_pos.to_excel(writer, sheet_name='位置', index=False, float_format="%.6f")
    df_vel.to_excel(writer, sheet_name='速度', index=False, float_format="%.6f")

# ---- 打印题目要求的摘要表格 ----
handle_indices_to_print = {
    "龙头": 0,
    "第1节龙身": 1,
    "第51节龙身": 51,
    "第101节龙身": 101,
    "第151节龙身": 151,
    "第201节龙身": 201,
    "龙尾(后)": 223
}
times_to_print = [0, 60, 120, 180, 240, 300]

# 创建用于打印的DataFrame
pos_table_data = {}
vel_table_data = {}
for t in times_to_print:
    time_idx = np.where(T_EVAL == t)[0][0]
    pos_col = []
    vel_col = []
    for name, handle_idx in handle_indices_to_print.items():
        pos_x = all_positions[time_idx, handle_idx, 0]
        pos_y = all_positions[time_idx, handle_idx, 1]
        speed = all_speeds[time_idx, handle_idx]
        pos_col.extend([f"{pos_x:.6f}", f"{pos_y:.6f}"])
        vel_col.append(f"{speed:.6f}")
    pos_table_data[f"{t} s"] = pos_col
    vel_table_data[f"{t} s"] = vel_col

row_labels_pos = []
row_labels_vel = []
for name in handle_indices_to_print.keys():
    row_labels_pos.extend([f"{name} x (m)", f"{name} y (m)"])
    row_labels_vel.append(f"{name} (m/s)")
    
df_pos_print = pd.DataFrame(pos_table_data, index=row_labels_pos)
df_vel_print = pd.DataFrame(vel_table_data, index=row_labels_vel)

print("\n" + "="*20 + " 仿真结果摘要 " + "="*20)
print("\n--- 表1: 关键把手在特定时刻的位置 (单位: m) ---")
print(df_pos_print)

print("\n--- 表2: 关键把手在特定时刻的速度大小 (单位: m/s) ---")
print(df_vel_print)

print("\n" + "="*58)
print(f"\n✅ 仿真完成，完整结果已保存到文件: {os.path.abspath(output_filename)}")
