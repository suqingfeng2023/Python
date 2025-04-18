#!/usr/bin/env python
# coding: utf-8

# In[16]:


from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import numpy as np
from matplotlib import rcParams

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['PingFang HK']  # 设置字体为PingFang HK

rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 输入数据
demand = {"梅苑1栋": 120, "菊苑1栋": 120}
supply = {"一食堂": 80, "二食堂": 114, "三食堂": 123, "教学2楼": 30, "教学4楼": 20, "计算机学院": 17}

# 距离矩阵（km）和运输时间（小时=距离/25km/h）
dist = {
    ("一食堂","梅苑1栋"):0.2, ("二食堂","梅苑1栋"):0.1, ("三食堂","梅苑1栋"):0.3,
    ("一食堂","菊苑1栋"):0.15,("二食堂","菊苑1栋"):0.5,("三食堂","菊苑1栋"):0.4,
    ("教学2楼","梅苑1栋"):1.0,("教学4楼","梅苑1栋"):0.9,("计算机学院","梅苑1栋"):0.8,
    ("教学2楼","菊苑1栋"):0.7,("教学4楼","菊苑1栋"):0.6,("计算机学院","菊苑1栋"):0.5
}




# 创建模型
prob = LpProblem("Bike_Rebalance_with_Trucks", LpMinimize)

# 决策变量
# x[i,j,k]: 调度车k从i到j运输的单车数量
trucks = [1, 2, 3]  # 3辆调度车
x = LpVariable.dicts("transport", 
                    [(i,j,k) for i in supply for j in demand for k in trucks],
                    lowBound=0, 
                    cat='Integer')

# y[i,j,k]: 是否使用调度车k从i到j（0-1变量）
y = LpVariable.dicts("use_truck",
                    [(i,j,k) for i in supply for j in demand for k in trucks],
                    cat='Binary')

# 目标函数：最小化总调度时间（最长单车的运输时间）
prob += lpSum(dist[(i,j)] * y[(i,j,k)] for i in supply for j in demand for k in trucks)

# 约束条件
# 1. 供给点不能调出超过存量
for i in supply:
    prob += lpSum(x[(i,j,k)] for j in demand for k in trucks) <= supply[i]

# 2. 需求点必须满足需求
for j in demand:
    prob += lpSum(x[(i,j,k)] for i in supply for k in trucks) >= demand[j]

# 3. 每车每次最多运输20辆
for k in trucks:
    for i in supply:
        for j in demand:
            prob += x[(i,j,k)] <= 20 * y[(i,j,k)]

# 4. 每辆车最多使用1次（简化假设）
for k in trucks:
    prob += lpSum(y[(i,j,k)] for i in supply for j in demand) <= 1

# 求解
prob.solve(PULP_CBC_CMD(msg=1))

# 输出结果
print("调度方案：")
for k in trucks:
    print(f"\n调度车 {k}:")
    for i in supply:
        for j in demand:
            if y[(i,j,k)].value() > 0.5:
                qty = x[(i,j,k)].value()
                print(f"  从 {i} 运输 {qty} 辆到 {j} (距离: {dist[(i,j)]}km, 耗时: {dist[(i,j)]/25*60:.1f}分钟)")

print(f"\n总调度时间: {prob.objective.value()*60:.1f} 分钟")


# In[17]:


import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
# 添加节点（不同颜色表示类型）
G.add_node("维修中心", type='depot')
for s in supply:
    G.add_node(s, type='supply') 
for d in demand:
    G.add_node(d, type='demand')

# 添加运输边
for (i,j,k), val in x.items():
    if val.value() > 0:
        G.add_edge(i, j, weight=dist[(i,j)], qty=val.value())

# 绘制
pos = nx.spring_layout(G)
colors = {'depot':'red', 'supply':'blue', 'demand':'green'}
nx.draw(G, pos, 
        node_color=[colors[G.nodes[n]['type']] for n in G.nodes],
        with_labels=True)
edge_labels = {(i,j):f"{G.edges[(i,j)]['qty']}辆" for i,j in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("共享单车调度网络")
plt.show()


# In[ ]:




