import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpBinary
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic


# ======================
# 1. 初始数据准备（修正版）
# ======================
def initialize_data():
    """初始化点位坐标、需求和限制路径（确认三食堂已存在）"""
    locations = {
        '运维处': (31.278, 121.519),
        '梅苑1栋': (31.275, 121.516),
        '菊苑1栋': (31.274, 121.515),
        '教2楼': (31.277, 121.520),
        '教4楼': (31.277, 121.521),
        '二食堂': (31.276, 121.517),
        '三食堂': (31.276, 121.518),  # 确认已存在
        '北门': (31.279, 121.518),
        '计算机学院': (31.278, 121.522),
        '体育馆': (31.275, 121.523)
    }

    demand = {
        '教2楼': 181,
        '教4楼': 123,
        '计算机学院': 44,
        '梅苑1栋': -9,
        '菊苑1栋': -13,
        '二食堂': -84,
        '三食堂': -40,  # 原需求值
        '体育馆': -34
    }

    restricted_paths = [('运维处', '体育馆'), ('北门', '计算机学院')]
    return locations, demand, restricted_paths


# ======================
# 2. 布局优化方案（修正版）
# ======================
def optimize_layout():
    """执行完整优化流程（修正三食堂处理）"""
    # 初始布局
    locations, demand, restricted_paths = initialize_data()
    dist_matrix = create_distance_matrix(locations, restricted_paths)
    schedule_df, _ = solve_scheduling(locations, demand, dist_matrix)
    original_eff = evaluate_efficiency(schedule_df, demand, num_vehicles=3, vehicle_capacity=20)

    # 布局调整方案（不再新增三食堂，改为调整现有点位）
    updated_demand = {
        '教2楼': 150,  # 分流部分需求到教5楼
        '教4楼': 100,  # 减少需求
        '三食堂': -60,  # 增加富余量（原-40）
        '计算机学院': 50,
        **{k: v for k, v in demand.items()
           if k not in ['教2楼', '教4楼', '三食堂', '计算机学院']}
    }

    # 新增教5楼点位（原无此点）
    updated_locations = {
        **locations,
        '教5楼': (31.277, 121.522)  # 仅新增教5楼
    }

    # 解除运维处->体育馆限制（保留其他限制）
    updated_restricted_paths = [('北门', '计算机学院')]

    # 重新求解
    updated_dist_matrix = create_distance_matrix(updated_locations, updated_restricted_paths)
    updated_schedule, _ = solve_scheduling(updated_locations, updated_demand, updated_dist_matrix)
    updated_eff = evaluate_efficiency(updated_schedule, updated_demand, num_vehicles=3, vehicle_capacity=20)

    return original_eff, updated_eff, schedule_df, updated_schedule


# ======================
# 3. 可视化对比（新增调度路线图）
# ======================
def plot_routes(schedule_df, locations, title):
    """绘制调度路线图"""
    G = nx.DiGraph()

    # 添加路线
    for _, row in schedule_df.iterrows():
        G.add_edge(row['出发地'], row['目的地'],
                   weight=row['运输量'],
                   vehicle=row['车辆'])

    # 坐标调整（经纬度转绘图坐标）
    pos = {loc: (coord[1], -coord[0]) for loc, coord in locations.items()}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # 分车辆绘制路线
    for k in schedule_df['车辆'].unique():
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['vehicle'] == k]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            width=1.5, edge_color=f'C{k - 1}',
            arrowsize=20, label=f'车辆{k}')

    plt.title(title, pad=20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ======================
# 主程序（修正版）
# ======================
if __name__ == "__main__":
    # 运行优化流程
    original_eff, updated_eff, original_schedule, updated_schedule = optimize_layout()

    # 打印结果对比
    print("\n=== 效率对比 ===")
    comp_df = pd.DataFrame([original_eff, updated_eff], index=['原布局', '新布局'])
    print(comp_df)

    # 可视化
    locations, _, _ = initialize_data()
    plot_results(original_eff, updated_eff)

    # 绘制调度路线图
    plot_routes(original_schedule, locations, "原布局调度路线")

    # 更新后的点位（新增教5楼）
    updated_locations = {**locations, '教5楼': (31.277, 121.522)}
    plot_routes(updated_schedule, updated_locations, "优化后调度路线")