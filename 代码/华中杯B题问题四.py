#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

# 定义停车点坐标（根据校园地图附件2进行适当估算，单位：米）
coord = {
    "维修中心": (1300, 50),  # 校园东北角运维处
    "梅苑1栋": (500, 1150), "菊苑1栋": (540, 850),
    "一食堂": (450, 950), "二食堂": (460, 1170), "三食堂": (350, 1000),
    "教学2楼": (800, 500), "教学4楼": (880, 600), "计算机学院": (720, 580),
    "工程中心": (1080, 540), "东门": (1130, 820), "南门": (700, 1650), "北门": (700,  100),
    "体育馆": (360, 550), "网球场": (530, 620), "校医院": (200, 600)
}
# 定义每日报修故障车辆分布（假设全天故障率6%，约47辆故障车）
# 假设故障车辆在中午和晚上各出现一半，并分布在当天最繁忙的区域
midday_broken = {"教学2楼": 8, "教学4楼": 6, "计算机学院": 3, "工程中心": 3}  # 中午教学区产生的故障车
evening_broken = {"梅苑1栋": 5, "菊苑1栋": 5, "一食堂": 2, "二食堂": 3, "三食堂": 3, "东门": 2, "南门": 1, "北门": 1}  # 晚上宿舍区及出入口的故障车

# 定义函数计算两点距离
def distance(p1, p2):
    x1,y1 = coord[p1]; x2,y2 = coord[p2]
    return math.hypot(x2-x1, y2-y1)

# 贪心算法规划巡检路线：每次从维修中心出发，选取尽可能多的故障车点巡逻，直至载满或无更多故障，再返回
def plan_route(broken_dict, capacity=20):
    route = ["维修中心"]
    load = 0
    current = "维修中心"
    # 每次选择距离当前点最近的有故障车的点
    while load < capacity and any(n>0 for n in broken_dict.values()):
        # 找最近的故障点
        next_point = min((p for p,v in broken_dict.items() if v>0), key=lambda p: distance(current,p))
        route.append(next_point)
        # 装载故障车
        if broken_dict[next_point] + load <= capacity:
            # 可以一次装完该点所有故障车
            load += broken_dict[next_point]
            broken_dict[next_point] = 0
        else:
            # 装满20辆即止
            broken_dict[next_point] -= (capacity - load)
            load = capacity
        current = next_point
        if load >= capacity:
            break
    route.append("维修中心")
    return route, load

# 规划中午巡检路线（假定鲁迪中午巡检一次）
midday_route, picked_mid = plan_route(midday_broken.copy(), capacity=20)
print("中午巡检路线:", " -> ".join(midday_route), f"，捡回故障车 {picked_mid} 辆")

# 规划傍晚巡检路线（假定鲁迪傍晚巡检，可能需两次往返）
evening_routes = []
broken_evening = evening_broken.copy()
total_picked = 0
while any(v>0 for v in broken_evening.values()) and total_picked < 40:  # 最多两趟
    route, picked = plan_route(broken_evening, capacity=20)
    evening_routes.append((route, picked))
    total_picked += picked
    if picked < 20:
        break  # 不满载则已捡完
for i, (route, picked) in enumerate(evening_routes, start=1):
    print(f"傍晚巡检路线{i}:", " -> ".join(route), f"，捡回故障车 {picked} 辆")



# In[2]:


import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['PingFang HK']  # 设置字体为PingFang HK

rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义停车点坐标（根据校园地图附件2进行适当估算，单位：米）
coord = {
    "维修中心": (1300, 50),  # 校园东北角运维处
    "梅苑1栋": (500, 1150), "菊苑1栋": (540, 850),
    "一食堂": (450, 950), "二食堂": (460, 1170), "三食堂": (350, 1000),
    "教学2楼": (800, 500), "教学4楼": (880, 600), "计算机学院": (720, 580),
    "工程中心": (1080, 540), "东门": (1130, 820), "南门": (700, 1650), "北门": (700,  100),
    "体育馆": (360, 550), "网球场": (530, 620), "校医院": (200, 600)
}

# 提取位置名称和坐标
names = list(coord.keys())
coordinates = np.array(list(coord.values()))

# 规划中午巡检路线（假定鲁迪中午巡检一次）
midday_route = ["维修中心", "教学2楼", "教学4楼", "计算机学院", "工程中心", "维修中心"]

# 规划傍晚巡检路线（假定鲁迪傍晚巡检，可能需两次往返）
evening_route1 = ["维修中心", "梅苑1栋", "菊苑1栋", "一食堂", "二食堂", "三食堂", "东门", "维修中心"]
evening_route2 = ["维修中心", "南门", "北门", "维修中心"]

# 创建一个图形
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制停车点坐标
ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', label='停车点')

# 标注停车点
for i, name in enumerate(names):
    ax.text(coordinates[i, 0], coordinates[i, 1], name, fontsize=9, ha='right', va='bottom')

# 绘制巡检路线
def plot_route(route, color):
    route_coords = np.array([coord[point] for point in route])
    ax.plot(route_coords[:, 0], route_coords[:, 1], color=color, marker='o', label='巡检路线')

plot_route(midday_route, 'green')
plot_route(evening_route1, 'orange')
plot_route(evening_route2, 'purple')

# 添加标题和标签
ax.set_title("校园巡检路线", fontsize=16)
ax.set_xlabel("X坐标 (米)")
ax.set_ylabel("Y坐标 (米)")

# 显示图例
ax.legend()

# 展示图形
plt.grid(True)
plt.show()


# In[ ]:




