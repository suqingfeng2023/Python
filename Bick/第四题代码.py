# -------------------------------
# 关键参数设置
# -------------------------------
total_bikes = 1771  # 总存量
failure_rate = 0.06  # 故障率
fault_bikes = total_bikes * failure_rate  # 故障车数量
max_capacity_per_trip = 20  # 每次修复车的最大运载能力
speed_per_min = 416.67  # 速度 (米/分钟)
loading_time_per_bike = 1  # 装卸时间 (分钟/辆)

# -------------------------------
# 各时段巡检数据
# -------------------------------
# 早高峰（7:00-9:00）
morning_bikes = 39  # 早高峰修复的故障车数
morning_time = 45.6  # 早高峰总耗时（分钟）

# 午间（11:00-13:00）
afternoon_bikes = 20  # 午间修复的故障车数
afternoon_time = 25.7  # 午间总耗时（分钟）

# 傍晚（17:00-19:00）
evening_bikes = 27  # 傍晚修复的故障车数
evening_time = 42.3  # 傍晚总耗时（分钟）

# -------------------------------
# 故障车数和剩余车数计算
# -------------------------------
# 总运回
total_bikes_returned = morning_bikes + afternoon_bikes + evening_bikes

# 总耗时
total_time = morning_time + afternoon_time + evening_time

# 剩余故障车
remaining_faults = round(fault_bikes - total_bikes_returned)

# 故障率计算
remaining_fault_rate = (remaining_faults / total_bikes) * 100

# 修复效率计算
repair_efficiency = (total_bikes_returned / fault_bikes) * 100

# 输出结果
print("结果分析：")
print(f"总修复故障车数: {total_bikes_returned}辆")
print(f"总耗时: {total_time:.2f}分钟")
print(f"修复效率: {repair_efficiency:.2f}%")
print(f"剩余故障车数: {remaining_faults}辆")
print(f"剩余故障率: {remaining_fault_rate:.2f}%")
