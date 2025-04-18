import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solution import pywrapcp

# 读取问题1的结果数据
bike_distribution = pd.read_csv('问题1结果表.csv', index_col=0)


# 用车需求模型
def calculate_demand(df):
    # 计算每个点位在每个时间点的需求变化率
    demand = pd.DataFrame(index=df.index, columns=df.columns)

    for i in range(len(df.columns) - 1):
        current_time = df.columns[i]
        next_time = df.columns[i + 1]
        time_diff = int(next_time.split(':')[0]) - int(current_time.split(':')[0])  # 小时差

        # 计算需求变化率 (简化处理)
        demand[current_time] = (df[next_time] - df[current_time]) / (time_diff if time_diff != 0 else 1)

    # 最后一个时间点设为0
    demand[df.columns[-1]] = 0

    return demand


# 计算需求
demand = calculate_demand(bike_distribution)
print("各停车点在不同时间的用车需求:")
print(demand)

# 高峰期识别 (根据附件3作息时间表)
peak_periods = {
    'morning': '9:00',  # 早高峰
    'noon': '12:00',  # 午高峰
    'evening': '18:00'  # 晚高峰
}


# 调度模型 (车辆路径问题VRP)
def create_distance_matrix(points):
    # 这里简化处理，实际应根据附件2校园地图计算各点之间的距离
    # 假设所有点位之间的距离是随机生成的(单位: km)
    num_points = len(points)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                # 随机距离 0.1-3km之间
                distance_matrix[i][j] = np.random.uniform(0.1, 3)

    return distance_matrix


# 调度参数
num_vehicles = 3
vehicle_capacity = 20  # 每辆车最多载20辆
vehicle_speed = 25  # km/h
depot = 0  # 假设运维处是第一个点位

# 创建距离矩阵
points = list(bike_distribution.index)
distance_matrix = create_distance_matrix(points)


# 计算调度量 (简化处理: 需求量大的点位需要补充车辆)
def calculate_supply(bike_dist, demand_df, peak_time):
    supply = {}
    for i, point in enumerate(points):
        # 如果当前存量低于平均且需求为正，则需要补充
        avg_bikes = bike_dist.mean(axis=1)[point]
        current_bikes = bike_dist[peak_time][point]

        if current_bikes < avg_bikes and demand_df[peak_time][point] > 0:
            supply[i] = min(vehicle_capacity, avg_bikes - current_bikes)
        else:
            supply[i] = 0

    return supply


# 早高峰调度示例
peak_time = peak_periods['morning']
supply = calculate_supply(bike_distribution, demand, peak_time)

print(f"\n{peak_time}时段的调度需求:")
for i, point in enumerate(points):
    print(f"{point}: {supply[i]}辆")


# 创建并求解VRP模型
def solve_vrp(distance_matrix, supply, num_vehicles, depot):
    # 创建数据模型
    data = {}
    data['distance_matrix'] = distance_matrix
    data['demands'] = [supply.get(i, 0) for i in range(len(points))]
    data['vehicle_capacities'] = [vehicle_capacity] * num_vehicles
    data['num_vehicles'] = num_vehicles
    data['depot'] = depot

    # 创建路由索引管理器
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # 创建路由模型
    routing = pywrapcp.RoutingModel(manager)

    # 创建距离回调函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 设置弧成本
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加容量约束
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # 设置搜索参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30

    # 求解问题
    solution = routing.SolveWithParameters(search_parameters)

    # 打印结果
    if solution:
        print("\n调度方案:")
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = f'调度车 {vehicle_id} 的路线:\n'
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += f' {points[node_index]} ->'
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += f' {points[manager.IndexToNode(index)]}\n'
            plan_output += f'运输单车数量: {route_load}\n'
            plan_output += f'路线距离: {route_distance}km\n'
            plan_output += f'预计时间: {route_distance / vehicle_speed * 60:.1f}分钟\n'
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print(f'总运输单车数量: {total_load}')
        print(f'总行驶距离: {total_distance}km')
    else:
        print("未找到解决方案")


# 求解VRP问题
solve_vrp(distance_matrix, supply, num_vehicles, depot)