import pandas as pd
import numpy as np
from datetime import datetime

# 读取附件1数据
data = pd.read_excel('附件1-共享单车分布统计表.xlsx', header=1)

# 数据预处理
# 将"200+"视为200，缺失值补0
data = data.fillna(0)

# 获取各点位在不同时间的单车数量
points = data.columns[2:-1]  # 排除前两列和最后一列

# 计算总量 - 最大值法
max_values = data[points].max()
total_bikes_max = max_values.sum()
print(f"共享单车总量(最大值法): {int(total_bikes_max)}辆")

# 计算总量 - 平均值法
mean_values = data[points].mean()
total_bikes_mean = mean_values.sum()
print(f"共享单车总量(平均值法): {int(round(total_bikes_mean))}辆")

# 选择更合理的估计值 - 这里选择最大值法
total_bikes = int(total_bikes_max)

# 统计各点位在指定时间点的分布
target_times = ['7:00', '9:00', '12:00', '14:00', '18:00', '21:00', '23:00']

# 创建结果表
result_table = pd.DataFrame(index=points, columns=target_times)


# 改进的时间匹配函数
def find_closest_time(row_time, target):
    # 将时间转换为分钟数进行比较
    def time_to_minutes(t):
        if pd.isna(t) or t == 0:
            return None
        if isinstance(t, str):
            try:
                h, m = map(int, t.split(':'))
                return h * 60 + m
            except:
                return None
        if hasattr(t, 'hour'):  # 如果是时间对象
            return t.hour * 60 + t.minute
        return None

    target_min = time_to_minutes(target)
    row_min = time_to_minutes(row_time)

    if row_min is None or target_min is None:
        return False

    # 允许±30分钟的窗口
    return abs(row_min - target_min) <= 30


# 填充结果表的改进方法
for point in points:
    for time in target_times:
        # 尝试直接匹配时间
        time_parts = list(map(int, time.split(':')))
        hour, minute = time_parts[0], time_parts[1]

        # 查找匹配的时间点
        matched_rows = []
        for idx, row in data.iterrows():
            row_time = row.iloc[0]
            if isinstance(row_time, str):
                try:
                    rt = datetime.strptime(row_time, '%H:%M:%S').time()
                    if rt.hour == hour and abs(rt.minute - minute) <= 30:
                        matched_rows.append(row)
                except:
                    continue
            elif hasattr(row_time, 'hour'):  # 如果是时间对象
                if row_time.hour == hour and abs(row_time.minute - minute) <= 30:
                    matched_rows.append(row)

        if matched_rows:
            # 取第一个匹配行的值
            value = matched_rows[0][point]
            result_table.loc[point, time] = value if not pd.isna(value) and value != 0 else 0
        else:
            # 如果没有匹配，使用该点位的平均值
            result_table.loc[point, time] = int(round(data[point].mean()))

# 确保所有值为整数
result_table = result_table.applymap(lambda x: int(x) if not pd.isna(x) else 0)

# 输出结果表
print("\n各停车点位在不同时间点的分布统计:")
print(result_table)

# 保存结果到CSV
result_table.to_csv('问题1结果表.csv')
print("\n结果已保存到'问题1结果表.csv'")