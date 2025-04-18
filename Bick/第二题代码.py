import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpBinary
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# ======================
# 1. 基于地图的精确数据准备
# ======================

locations = {
    '运维处': (31.278, 121.519),
    '梅苑1栋': (31.275, 121.516),
    '菊苑1栋': (31.274, 121.515),
    '教2楼': (31.277, 121.520),
    '教4楼': (31.277, 121.521),
    '二食堂': (31.276, 121.517),
    '北门': (31.279, 121.518),
    '计算机学院': (31.278, 121.522),
    '体育馆': (31.275, 121.523)
}

# 早高峰需求数据
demand = {
    '教2楼': 181,
    '教4楼': 123,
    '计算机学院': 44,
    '梅苑1栋': -9,
    '菊苑1栋': -13,
    '二食堂': -84,
    '体育馆': -34
}

# 禁止通行路径
restricted_paths = [('运维处', '体育馆'), ('北门', '计算机学院')]

# ======================
# 2. 精确距离矩阵计算
# ======================

def create_distance_matrix(loc_dict, restrictions):
    nodes = list(loc_dict.keys())
    dist_matrix = pd.DataFrame(np.inf, index=nodes, columns=nodes)

    for src in nodes:
        for dst in nodes:
            if src == dst:
                dist_matrix.loc[src, dst] = 0
            elif (src, dst) in restrictions:
                continue
            else:
                dist_matrix.loc[src, dst] = geodesic(loc_dict[src], loc_dict[dst]).km

    return dist_matrix

dist_matrix = create_distance_matrix(locations, restricted_paths)

# ======================
# 3. 增强的优化模型
# ======================

model = LpProblem("Enhanced_Bike_Scheduling", LpMinimize)

# 参数设置
num_vehicles = 3
vehicle_capacity = 20
vehicle_speed = 25  # km/h
max_operating_time = 90  # 分钟
loading_time = 3  # 分钟/次

# 创建决策变量
nodes = list(locations.keys())
x = LpVariable.dicts("Route",
                     [(i, j, k) for i in nodes
                      for j in nodes
                      for k in range(num_vehicles)
                      if i != j and not np.isinf(dist_matrix.loc[i, j])],
                     cat=LpBinary)

# 辅助变量：未满足需求
unmet = LpVariable.dicts("Unmet", [j for j in nodes if demand.get(j, 0) > 0], lowBound=0)

# 目标函数：最小化未满足需求和调度成本
model += lpSum(unmet[j] for j in unmet) + 0.01 * lpSum(x[(i, j, k)] for (i, j, k) in x)

# 约束条件
# 1. 需求满足约束
for j in [n for n in nodes if demand.get(n, 0) > 0]:
    model += vehicle_capacity * lpSum(x[(i, j, k)] for i in nodes for k in range(num_vehicles)
                                      if i != j and (i, j, k) in x) + unmet[j] >= demand[j]

# 2. 车辆工作时间约束
for k in range(num_vehicles):
    model += lpSum(x[(i, j, k)] * (dist_matrix.loc[i, j] / vehicle_speed * 60 + loading_time)
                   for (i, j, k) in x) <= max_operating_time

# 3. 节点流量平衡
for i in nodes:
    for k in range(num_vehicles):
        model += lpSum(x[(i, j, k)] for j in nodes if (i, j, k) in x) <= 1  # 每车每节点最多出发一次
        model += lpSum(x[(j, i, k)] for j in nodes if (j, i, k) in x) <= 1  # 每车每节点最多到达一次

# 4. 富余点输出限制
for i in [n for n in nodes if demand.get(n, 0) < 0]:
    model += lpSum(x[(i, j, k)] for j in nodes for k in range(num_vehicles)
                   if (i, j, k) in x) <= -demand[i] / vehicle_capacity

# ======================
# 4. 模型求解与结果分析
# ======================
model.solve()

print("\n求解状态:", LpStatus[model.status])

# 结果整理
schedule = []
for (i, j, k) in x:
    if x[(i, j, k)].varValue > 0.9:  # 考虑浮点误差
        travel_time = dist_matrix.loc[i, j] / vehicle_speed * 60
        # 计算出发时间
        start_time = 6 * 60 + 30 + k * 20  # 转换为分钟
        hours = start_time // 60
        minutes = start_time % 60
        formatted_time = f"{hours}:{minutes:02d}"

        schedule.append({
            '车辆': k + 1,
            '出发地': i,
            '目的地': j,
            '出发时间': formatted_time,
            '运输量': vehicle_capacity,
            '耗时(分钟)': round(travel_time + loading_time)
        })

# 转换为DataFrame输出
schedule_df = pd.DataFrame(schedule)
print("\n最优调度方案:")
print(schedule_df.sort_values(['车辆', '出发时间']))

# 计算关键指标
total_unmet = sum(unmet[j].varValue for j in unmet)
fulfillment_rate = 1 - total_unmet / sum(demand[j] for j in demand if j in unmet)
print(f"\n需求满足率: {fulfillment_rate:.1%}")
print(f"总调度车次: {len(schedule_df)}")
print(f"平均单车利用率: {len(schedule_df) / num_vehicles:.1f} 趟/车")

# ======================
# 5. 可视化调度网络
# ======================
G = nx.DiGraph()

# 添加节点和边
for _, row in schedule_df.iterrows():
    G.add_edge(row['出发地'], row['目的地'],
               weight=row['运输量'],
               vehicle=row['车辆'],
               time=row['耗时(分钟)'])

# 绘制
plt.figure(figsize=(12, 8))
pos = {loc: (coord[1], -coord[0]) for loc, coord in locations.items()}  # 经纬度转绘图坐标

# 绘制基础网络
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10)

# 绘制带权重的边
for (u, v, d) in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                           width=d['weight'] / 10,
                           arrowsize=20,
                           edge_color=f"C{d['vehicle']}",
                           label=f"车{d['vehicle']}")

# 添加图例和标题
plt.legend(title='调度车编号')
plt.title("校园共享单车最优调度路线\n(线宽表示运输量，颜色区分车辆)", pad=20)
plt.grid(True)
plt.tight_layout()
plt.show()
