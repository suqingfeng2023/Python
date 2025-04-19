#!/usr/bin/env python
# coding: utf-8

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import rcParams

# 设置中文字体显示
rcParams['font.sans-serif'] = ['PingFang HK']
rcParams['axes.unicode_minus'] = False


# 调度函数
def optimize_dispatch(supply, demand, distances, title):
    model = LpProblem("Shared_Bike_Dispatch", LpMinimize)
    vehicles = [1, 2, 3]

    transport_quantity = LpVariable.dicts("transport",
                                          [(i, j, k) for i in supply for j in demand for k in vehicles],
                                          lowBound=0, cat='Integer')

    use_truck_indicator = LpVariable.dicts("use_truck",
                                           [(i, j, k) for i in supply for j in demand for k in vehicles],
                                           cat='Binary')

    # 🚛 最小化总运输时间（单位：分钟）
    model += lpSum(distances[(i, j)] / 25 * 60 * use_truck_indicator[(i, j, k)]
                   for i in supply for j in demand for k in vehicles)

    # 每个供给点不能调出超过其存量
    for i in supply:
        model += lpSum(transport_quantity[(i, j, k)] for j in demand for k in vehicles) <= supply[i]

    # 每个需求点必须满足需求
    for j in demand:
        model += lpSum(transport_quantity[(i, j, k)] for i in supply for k in vehicles) >= demand[j]

    # 每车每次最多运20辆
    for k in vehicles:
        for i in supply:
            for j in demand:
                model += transport_quantity[(i, j, k)] <= 20 * use_truck_indicator[(i, j, k)]

    # 每辆车最多使用一次
    for k in vehicles:
        model += lpSum(use_truck_indicator[(i, j, k)] for i in supply for j in demand) <= 1

    # 求解模型
    model.solve(PULP_CBC_CMD(msg=0))

    print(f"\n🚲🚚 {title} 调度方案")
    for k in vehicles:
        used = False
        for i in supply:
            for j in demand:
                if use_truck_indicator[(i, j, k)].value() > 0.5:
                    used = True
                    qty = transport_quantity[(i, j, k)].value()
                    time = distances[(i, j)] / 25 * 60
                    print(f"  🛻 车{k}：从 {i} 运送 {int(qty)} 辆到 {j}（距离 {distances[(i, j)]}km，用时 {time:.1f} 分钟）")
        if not used:
            print(f"  🛻 车{k}：未出车")

    total_time = model.objective.value()
    print(f"⏱️ 总调度时间（估算最长路径）：{total_time:.1f} 分钟")

    # 可视化调度网络图
    G = nx.DiGraph()
    for node in supply: G.add_node(node, type='supply')
    for node in demand: G.add_node(node, type='demand')

    for (i, j, k), var in transport_quantity.items():
        if var.value() > 0:
            G.add_edge(i, j, weight=distances[(i, j)], qty=var.value())

    pos = nx.spring_layout(G, seed=42)
    colors = {'supply': 'skyblue', 'demand': 'lightgreen'}
    nx.draw(G, pos,
            node_color=[colors[G.nodes[n]['type']] for n in G.nodes],
            with_labels=True, node_size=1000, font_size=10)
    edge_labels = {(i, j): f"{int(G.edges[i, j]['qty'])}辆" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title + " 调度网络")
    plt.show()


# ========== 三个时段的数据 ==========

# 早高峰（宿舍为需求）
supply_morning = {
    "一食堂": 50, "二食堂": 60, "三食堂": 40, "教学2楼": 30
}
demand_morning = {
    "梅苑1栋": 70, "菊苑1栋": 80
}
dist_morning = {
    ("一食堂", "梅苑1栋"): 0.3, ("二食堂", "梅苑1栋"): 0.4, ("三食堂", "梅苑1栋"): 0.5, ("教学2楼", "梅苑1栋"): 0.6,
    ("一食堂", "菊苑1栋"): 0.4, ("二食堂", "菊苑1栋"): 0.5, ("三食堂", "菊苑1栋"): 0.6, ("教学2楼", "菊苑1栋"): 0.7
}

# 午高峰（教学区为需求）
supply_noon = {
    "一食堂": 60, "二食堂": 50, "三食堂": 40
}
demand_noon = {
    "教学2楼": 80, "教学4楼": 70
}
dist_noon = {
    ("一食堂", "教学2楼"): 0.4, ("二食堂", "教学2楼"): 0.5, ("三食堂", "教学2楼"): 0.6,
    ("一食堂", "教学4楼"): 0.5, ("二食堂", "教学4楼"): 0.6, ("三食堂", "教学4楼"): 0.7
}

# 晚高峰（宿舍为供给）
supply_night = {
    "梅苑1栋": 50, "菊苑1栋": 60
}
demand_night = {
    "教学2楼": 40, "教学4楼": 50, "计算机学院": 30
}
dist_night = {
    ("梅苑1栋", "教学2楼"): 0.6, ("菊苑1栋", "教学2楼"): 0.7,
    ("梅苑1栋", "教学4楼"): 0.7, ("菊苑1栋", "教学4楼"): 0.8,
    ("梅苑1栋", "计算机学院"): 0.5, ("菊苑1栋", "计算机学院"): 0.6
}

# ========== 执行三次调度 ==========
optimize_dispatch(supply_morning, demand_morning, dist_morning, "早高峰（上学）")
optimize_dispatch(supply_noon, demand_noon, dist_noon, "午高峰（吃饭）")
optimize_dispatch(supply_night, demand_night, dist_night, "晚高峰（归宿）")
