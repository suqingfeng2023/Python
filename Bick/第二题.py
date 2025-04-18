#!/usr/bin/env python
# coding: utf-8

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import numpy as np
from matplotlib import rcParams
import networkx as nx
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['PingFang HK']  # 设置字体为PingFang HK
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 输入数据
supply_points = {"一食堂": 80, "二食堂": 114, "三食堂": 123, "教学2楼": 30, "教学4楼": 20, "计算机学院": 17}
demand_points = {"梅苑1栋": 120, "菊苑1栋": 120}

# 距离矩阵（km）和运输时间（小时=距离/25km/h）
distances = {
    ("一食堂","梅苑1栋"):0.2, ("二食堂","梅苑1栋"):0.1, ("三食堂","梅苑1栋"):0.3,
    ("一食堂","菊苑1栋"):0.15,("二食堂","菊苑1栋"):0.5,("三食堂","菊苑1栋"):0.4,
    ("教学2楼","梅苑1栋"):1.0,("教学4楼","梅苑1栋"):0.9,("计算机学院","梅苑1栋"):0.8,
    ("教学2楼","菊苑1栋"):0.7,("教学4楼","菊苑1栋"):0.6,("计算机学院","菊苑1栋"):0.5
}

# 创建优化模型
model = LpProblem("Bike_Rebalance_with_Truck_Scheduling", LpMinimize)

# 定义决策变量
# transport_quantity[i,j,k]: 从i到j通过调度车k运输的单车数量
vehicles = [1, 2, 3]  # 3辆调度车
transport_quantity = LpVariable.dicts("transport",
                    [(i,j,k) for i in supply_points for j in demand_points for k in vehicles],
                    lowBound=0,
                    cat='Integer')

# use_truck_indicator[i,j,k]: 是否使用调度车k从i到j（0-1变量）
use_truck_indicator = LpVariable.dicts("use_truck",
                    [(i,j,k) for i in supply_points for j in demand_points for k in vehicles],
                    cat='Binary')

# 目标函数：最小化总调度时间（最长运输时间）
model += lpSum(distances[(i,j)] * use_truck_indicator[(i,j,k)] for i in supply_points for j in demand_points for k in vehicles)

# 约束条件
# 1. 供给点不能调出超过存量
for i in supply_points:
    model += lpSum(transport_quantity[(i,j,k)] for j in demand_points for k in vehicles) <= supply_points[i]

# 2. 需求点必须满足需求
for j in demand_points:
    model += lpSum(transport_quantity[(i,j,k)] for i in supply_points for k in vehicles) >= demand_points[j]

# 3. 每车每次最多运输20辆
for k in vehicles:
    for i in supply_points:
        for j in demand_points:
            model += transport_quantity[(i,j,k)] <= 20 * use_truck_indicator[(i,j,k)]

# 4. 每辆车最多使用一次（简化假设）
for k in vehicles:
    model += lpSum(use_truck_indicator[(i,j,k)] for i in supply_points for j in demand_points) <= 1

# 求解模型
model.solve(PULP_CBC_CMD(msg=1))

# 输出调度结果
print("调度方案：")
for k in vehicles:
    print(f"\n调度车 {k}:")
    for i in supply_points:
        for j in demand_points:
            if use_truck_indicator[(i,j,k)].value() > 0.5:
                qty = transport_quantity[(i,j,k)].value()
                print(f"  从 {i} 运输 {qty} 辆到 {j} (距离: {distances[(i,j)]}km, 耗时: {distances[(i,j)]/25*60:.1f}分钟)")

print(f"\n总调度时间: {model.objective.value()*60:.1f} 分钟")

# 可视化调度网络
G = nx.DiGraph()
# 添加节点（不同颜色表示类型）
G.add_node("维修中心", type='depot')
for s in supply_points:
    G.add_node(s, type='supply')
for d in demand_points:
    G.add_node(d, type='demand')

# 添加运输边
for (i,j,k), val in transport_quantity.items():
    if val.value() > 0:
        G.add_edge(i, j, weight=distances[(i,j)], qty=val.value())

# 绘制调度网络图
pos = nx.spring_layout(G)
colors = {'depot':'red', 'supply':'blue', 'demand':'green'}
nx.draw(G, pos,
        node_color=[colors[G.nodes[n]['type']] for n in G.nodes],
        with_labels=True)
edge_labels = {(i,j):f"{G.edges[(i,j)]['qty']}辆" for i,j in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("共享单车调度网络")
plt.show()
