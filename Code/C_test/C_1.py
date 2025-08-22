# 导入所需的库
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

def solve_crop_optimization(scenario):
    """
    解决农作物种植优化问题的主函数。
    参数:
    scenario (int): 1 代表情景1.1 (超产滞销), 2 代表情景1.2 (超产半价销售)。
    """
    # --- 1. 数据加载 ---
    try:
        df_land = pd.read_csv('附件1.xlsx - 乡村的现有耕地.csv', encoding='gbk')
        df_crops_info = pd.read_csv('附件1.xlsx - 乡村种植的农作物.csv', encoding='gbk')
        df_history_2023 = pd.read_csv('附件2.xlsx - 2023年的农作物种植情况.csv', encoding='gbk')
        df_stats_2023 = pd.read_csv('附件2.xlsx - 2023年统计的相关数据.csv', encoding='gbk')
    except Exception as e:
        print(f"文件读取失败，请确保所有CSV文件都在同一目录下，并检查文件编码。错误: {e}")
        return None, None

    # --- 2. 数据预处理和参数计算 ---

    # a. 地块信息
    land_info = df_land.set_index('地块名称')['地块面积/亩'].to_dict()
    land_types = df_land.set_index('地块名称')['地块类型'].to_dict()
    plots = list(land_info.keys())

    # b. 作物信息
    crops = df_crops_info['作物名称'].tolist()
    crop_ids = df_crops_info.set_index('作物名称')['作物编号'].to_dict()
    crop_types = df_crops_info.set_index('作物名称')['作物类型'].to_dict()
    bean_crops = [c for c, t in crop_types.items() if '豆类' in t]

    # c. 经济参数 (成本, 产量, 价格)
    # 处理价格范围，取中值
    df_stats_2023['销售单价/(元/斤)'] = df_stats_2023['销售单价/(元/斤)'].astype(str).apply(
        lambda x: (float(x.split('-')[0]) + float(x.split('-')[1])) / 2 if '-' in x else float(x)
    )
    # 创建参数字典
    costs = {}
    yields = {}
    prices = {}
    for _, row in df_stats_2023.iterrows():
        crop_name = row['作物名称']
        land_type = row['地块类型']
        # 将参数与(作物, 地块类型)关联
        costs[(crop_name, land_type)] = row['种植成本/(元/亩)']
        yields[(crop_name, land_type)] = row['亩产量/斤']
        if crop_name not in prices:
            prices[crop_name] = row['销售单价/(元/斤)']

    # d. 计算2023年总产量作为预期销售量 (Demand)
    demands = {c: 0 for c in crops}
    for _, row in df_history_2023.iterrows():
        crop_name = row['作物名称']
        plot_name = row['种植地块']
        area = row['种植面积/亩']
        # 处理多季作物的情况
        if pd.isna(plot_name) or pd.isna(crop_name):
            continue
        land_type = land_types[plot_name]
        if (crop_name, land_type) in yields:
            demands[crop_name] += area * yields[(crop_name, land_type)]

    # e. 2023年种植历史
    history_2023 = {plot: [] for plot in plots}
    # 清理数据，去除无效行
    df_history_2023_cleaned = df_history_2023.dropna(subset=['种植地块', '作物名称'])
    for _, row in df_history_2023_cleaned.iterrows():
        history_2023[row['种植地块']].append(row['作物名称'])

    # f. 定义作物适宜性
    # 这是一个简化的适宜性规则，实际模型中会更复杂
    suitability = {}
    # ... 此处应根据附件1中的详细文字规则，构建一个复杂的适宜性判断字典或函数
    # 为简化示例，我们假设所有在df_stats_2023中出现的(作物,地块类型)组合都是适宜的
    for crop, land_type in costs.keys():
        if crop not in suitability:
            suitability[crop] = []
        if land_type not in suitability[crop]:
            suitability[crop].append(land_type)


    # --- 3. 建立优化模型 ---
    years = range(2024, 2031)
    seasons = [1, 2]

    # a. 创建问题实例
    model = LpProblem(f"Crop_Planning_Scenario_{scenario}", LpMaximize)

    # b. 定义决策变量
    # x_itcs: t年i地块s季种植作物c的面积
    x = LpVariable.dicts("Area", (plots, years, seasons, crops), lowBound=0, cat='Continuous')
    # z_itc: t年i地块是否种植了作物c
    z = LpVariable.dicts("IsPlanted", (plots, years, crops), cat='Binary')

    # c. 定义目标函数
    # 辅助变量：总产量、总成本、总收入
    total_production = {(c, t): lpSum(x[i][t][s][c] * yields.get((c, land_types[i]), 0)
                                      for i in plots for s in seasons
                                      if (c, land_types[i]) in yields)
                        for c in crops for t in years}

    total_cost = lpSum(x[i][t][s][c] * costs.get((c, land_types[i]), 0)
                       for i in plots for t in years for s in seasons for c in crops
                       if (c, land_types[i]) in costs)

    if scenario == 1: # 情景1.1: 超产滞销
        # 正常销售量 S_tc <= D_c, S_tc <= Q_tc
        sales_normal = LpVariable.dicts("SalesNormal", (crops, years), lowBound=0)
        for t in years:
            for c in crops:
                model += sales_normal[c][t] <= demands[c]
                model += sales_normal[c][t] <= total_production[(c, t)]
        total_revenue = lpSum(sales_normal[c][t] * prices[c] for c in crops for t in years)

    elif scenario == 2: # 情景1.2: 超产半价
        # 正常销售 S_tc, 超产销售 E_tc
        sales_normal = LpVariable.dicts("SalesNormal", (crops, years), lowBound=0)
        sales_excess = LpVariable.dicts("SalesExcess", (crops, years), lowBound=0)
        for t in years:
            for c in crops:
                model += sales_normal[c][t] + sales_excess[c][t] == total_production[(c, t)]
                model += sales_normal[c][t] <= demands[c]
        total_revenue = lpSum(sales_normal[c][t] * prices[c] + sales_excess[c][t] * prices[c] * 0.5
                              for c in crops for t in years)

    model += total_revenue - total_cost, "Total_Profit"

    # d. 添加约束条件
    # (1) 面积约束
    for i in plots:
        for t in years:
            for s in seasons:
                model += lpSum(x[i][t][s][c] for c in crops) <= land_info[i]

    # (2) 逻辑关联约束: 种植面积 > 0 则 isPlanted = 1
    M = 2000 # 一个足够大的数
    for i in plots:
        for t in years:
            for c in crops:
                model += lpSum(x[i][t][s][c] for s in seasons) <= M * z[i][t][c]
                model += z[i][t][c] <= lpSum(x[i][t][s][c] for s in seasons)


    # (3) 禁止重茬约束
    for i in plots:
        for c in crops:
            # 2024年不能种2023年种过的
            if c in history_2023[i]:
                model += z[i][2024][c] == 0
            # 2025年及以后
            for t in range(2025, 2031):
                model += z[i][t][c] + z[i][t-1][c] <= 1

    # (4) 豆类强制种植约束
    for i in plots:
        # 检查2023年是否种植豆类
        bean_in_2023 = any(c in bean_crops for c in history_2023[i])
        for t in range(2024, 2029): # 检查到2028年，以覆盖2030年
            if t == 2024:
                if not bean_in_2023: # 假设2022年也没种
                     model += lpSum(z[i][t][c] for c in bean_crops) + lpSum(z[i][t+1][c] for c in bean_crops) >= 1
            elif t == 2025:
                 if bean_in_2023: # 23年种了，24,25,26三年内必须种
                     model += lpSum(z[i][t-1][c] for c in bean_crops) + lpSum(z[i][t][c] for c in bean_crops) + lpSum(z[i][t+1][c] for c in bean_crops) >= 1
                 else: # 23年没种，24,25必须种
                     pass # 已在t=2024中约束
            else: # t >= 2026
                model += lpSum(z[i][t-2][c] for c in bean_crops) + lpSum(z[i][t-1][c] for c in bean_crops) + lpSum(z[i][t][c] for c in bean_crops) >= 1

    # (5) 适宜性约束 (简化版)
    for i in plots:
        land_type = land_types[i]
        for t in years:
            for s in seasons:
                for c in crops:
                    if c not in suitability or land_type not in suitability[c]:
                        model += x[i][t][s][c] == 0

    # (6) 特定地块规则 (简化版)
    # A, B, C类地块每年只能种一季
    for i in plots:
        if land_types[i] in ['平旱地', '梯田', '山坡地']:
            for t in years:
                model += lpSum(z[i][t][c] for c in crops) <= 1
                # 假设都种在第一季
                model += lpSum(x[i][t][2][c] for c in crops) == 0

    # ... 此处应添加更多关于 D, E, F 类地块的详细约束

    # --- 4. 求解模型 ---
    print(f"--- 开始求解情景 {scenario} ---")
    model.solve()
    print(f"求解状态: {LpStatus[model.status]}")

    # --- 5. 结果解析 ---
    if LpStatus[model.status] == 'Optimal':
        total_profit = model.objective.value()
        print(f"最优总利润: {total_profit:,.2f} 元")

        # 将结果格式化为DataFrame
        results = {}
        for t in years:
            # 创建一个空的DataFrame，索引为地块和季次，列为作物
            header = pd.MultiIndex.from_product([plots, seasons], names=['地块名', '季次'])
            df_year = pd.DataFrame(0.0, index=header, columns=crops)
            for i in plots:
                for s in seasons:
                    for c in crops:
                        val = x[i][t][s][c].value()
                        if val > 0.01: # 只记录有意义的种植面积
                            df_year.loc[(i, s), c] = val
            results[t] = df_year.reset_index()
        return total_profit, results
    else:
        print("未能找到最优解。")
        return None, None


if __name__ == '__main__':
    # 解决情景1.1
    profit_1, results_1 = solve_crop_optimization(1)

    # 解决情景1.2
    profit_2, results_2 = solve_crop_optimization(2)

    # 你可以在这里添加代码，将 results_1 和 results_2 中的DataFrame保存为要求的Excel格式
    # 例如，对于2024年的结果：
    # if results_1:
    #     results_1[2024].to_excel("result1_1_2024_output.xlsx", index=False)
    # if results_2:
    #     results_2[2025].to_excel("result1_2_2025_output.xlsx", index=False)
