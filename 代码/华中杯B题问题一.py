#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# 读取附件1数据（已整理为CSV格式）
data = pd.read_excel("附件1-共享单车分布统计表.xlsx", sheet_name="Sheet1", header=1)

# 数据清洗：处理“200+”和缺失值
data = data.replace("200+", 200).fillna(0)

# 提取所有停车点位名称
locations = data.columns[2:]  # 跳过前两列（日期和时间）

# 方法1：最大值法估算总量
total_bikes_max = data[locations].max().sum()

# 方法2：平均值法估算总量
total_bikes_mean = data[locations].mean().sum().round()

print(f"共享单车总量估算（最大值法）: {total_bikes_max} 辆")
print(f"共享单车总量估算（平均值法）: {total_bikes_mean} 辆")


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time

# 1. 正确读取数据
def load_data_correctly():
    # 先查看原始文件结构
    raw = pd.read_excel("附件1-共享单车分布统计表.xlsx", header=None)
    print("原始文件前3行：")
    print(raw.head(3))
    
    # 确定表头行（通常第2行是列名）
    header_row = 1
    
    # 重新读取数据，指定header行
    df = pd.read_excel("附件1-共享单车分布统计表.xlsx", header=header_row)
    
    # 清理数据：处理"200+"和空值
    df = df.replace("200+", 200).fillna(0)
    
    # 查找时间列（包含时间数据的列）
    time_col = None
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if any(isinstance(x, time) for x in sample) or any(":" in str(x) for x in sample):
            time_col = col
            break
    
    if not time_col:
        time_col = df.columns[1]  # 默认第二列为时间列
    
    print(f"\n使用时间列: {time_col}")
    
    # 标准化时间格式
    df['时间'] = pd.to_datetime(df[time_col].astype(str)).dt.time
    
    # 识别停车点位列（应该是表头的第一行）
    location_cols = [col for col in df.columns 
                    if col not in [time_col, '时间', df.columns[0]] 
                    and not str(col).startswith('Unnamed')]
    
    # 如果没有识别到，则使用除时间列外的所有数值列
    if not location_cols:
        location_cols = [col for col in df.columns 
                        if pd.api.types.is_numeric_dtype(df[col]) 
                        and col not in [time_col, '时间']]
    
    print(f"识别到停车点位: {location_cols}")
    
    return df, location_cols

# 2. 数据分析和可视化
def analyze_and_visualize(df, locations):
    # 计算各点位最大存量
    max_counts = df[locations].max().sort_values(ascending=False)
    
    # 计算总量（最大值法）
    total_bikes = max_counts.sum()
    print(f"\n校园共享单车总量（最大值法）: {int(total_bikes)} 辆")
    
    # 可视化各点位最大存量
    plt.figure(figsize=(14, 7))
    max_counts.plot(kind='bar', color='steelblue')
    plt.title('各停车点位最大单车存量分布', fontsize=14)
    plt.ylabel('单车数量', fontsize=12)
    plt.xlabel('停车点位', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('max_bikes_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 时间趋势分析
    df['小时'] = df['时间'].apply(lambda x: x.hour)
    hourly_avg = df.groupby('小时')[locations].mean()
    
    plt.figure(figsize=(15, 8))
    for loc in locations[:5]:  # 只显示前5个点位
        plt.plot(hourly_avg.index, hourly_avg[loc], label=loc, marker='o')
    
    plt.title('单车数量随时间变化趋势（前5个点位）', fontsize=14)
    plt.xlabel('时间（小时）', fontsize=12)
    plt.ylabel('平均单车数量', fontsize=12)
    plt.xticks(range(7, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('time_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 执行分析
try:
    print("开始数据分析...")
    df, locations = load_data_correctly()
    
    print("\n处理后的数据前5行：")
    print(df.head())
    
    analyze_and_visualize(df, locations)
    print("\n分析完成！图表已保存为PNG文件")
    
except Exception as e:
    print(f"\n分析过程中出错: {str(e)}")
    print("建议检查：")
    print("1. Excel文件路径是否正确")
    print("2. 文件是否被其他程序占用")
    print("3. 尝试手动指定header_row参数")


# In[12]:


import pandas as pd
import numpy as np
from datetime import time

def load_and_preprocess():
    # 读取原始数据（不自动解析时间）
    raw_data = pd.read_excel("附件1-共享单车分布统计表.xlsx", header=[0,1], dtype=str)
    
    # 重建单级列名（确保所有值为字符串）
    new_columns = []
    for col in raw_data.columns:
        part1 = str(col[0]) if not pd.isna(col[0]) else ""
        part2 = str(col[1]) if not pd.isna(col[1]) else ""
        new_columns.append(f"{part1} {part2}".strip())
    
    raw_data.columns = new_columns
    
    # 提取时间列（第二列）
    time_col = raw_data.columns[1]
    data = pd.DataFrame()
    data['时间'] = pd.to_datetime(raw_data[time_col], errors='coerce').dt.time
    
    # 标准点位列表（按题目要求顺序）
    standard_locations = [
        '东门', '南门', '北门', '一食堂', '二食堂', '三食堂',
        '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼',
        '计算机学院', '工程中心', '网球场', '体育馆', '校医院'
    ]
    
    # 匹配数据列
    for loc in standard_locations:
        # 查找包含点位名称的列
        matched_cols = [col for col in raw_data.columns if loc in col and col != time_col]
        if matched_cols:
            data[loc] = pd.to_numeric(raw_data[matched_cols[0]].replace("200+", 200), errors='coerce').fillna(0)
        else:
            data[loc] = 0
            print(f"警告: 未找到点位 {loc} 的对应数据列")
    
    # 添加时间辅助列
    data['总分钟'] = data['时间'].apply(lambda x: x.hour * 60 + x.minute)
    
    return data, standard_locations

def fill_table1(data, locations):
    # 定义目标时间点
    target_times = {
        '7:00': 7*60,
        '9:00': 9*60,
        '12:00': 12*60,
        '14:00': 14*60,
        '18:00': 18*60,
        '21:00': 21*60,
        '23:00': 23*60
    }
    
    result = pd.DataFrame(index=locations, columns=target_times.keys())
    
    for time_str, target_min in target_times.items():
        # 找到最接近的记录
        idx = (data['总分钟'] - target_min).abs().idxmin()
        
        # 填充数据
        for loc in locations:
            result.at[loc, time_str] = int(data.at[idx, loc])
    
    return result

# 主程序
try:
    print("正在处理数据...")
    data, locations = load_and_preprocess()
    
    print("\n处理后的数据样例:")
    print(data.head())
    
    print("\n正在生成表1结果...")
    table1 = fill_table1(data, locations)
    
    # 输出结果
    print("\n表1 共享单车分布统计结果:")
    print(table1)
    
    # 保存结果
    table1.to_excel('表1_共享单车分布统计结果.xlsx')
    print("\n结果已保存到: 表1_共享单车分布统计结果.xlsx")
    
    # 计算总量（最大值法）
    total_bikes = data[locations].max().sum()
    print(f"\n校园共享单车总量（最大值法）: {int(total_bikes)} 辆")

except Exception as e:
    print(f"\n处理过程中出错: {str(e)}")
    print("建议检查:")
    print("1. Excel文件是否与示例格式完全一致")
    print("2. 是否安装了最新版本的pandas库")


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# 数据准备
data = {
    '7:00': [31,47,15,0,91,0,0,106,0,26,4,0,0,0,0],
    '9:00': [0,0,31,54,0,47,106,93,0,34,8,0,37,0,6],
    '12:00': [86,52,0,0,58,66,0,66,50,0,0,38,0,14,0],
    '14:00': [43,0,66,10,0,0,19,62,0,136,0,70,19,0,35],
    '18:00': [36,99,0,0,80,65,0,0,33,0,59,0,0,45,0],
    '21:00': [85,0,46,55,0,0,119,118,31,0,7,43,2,0,1],
    '23:00': [0,41,0,85,122,0,113,0,0,22,17,0,16,0,13]
}
locations = ['东门','南门','北门','一食堂','二食堂','三食堂','梅苑1栋','菊苑1栋',
            '教学2楼','教学4楼','计算机学院','工程中心','网球场','体育馆','校医院']

df = pd.DataFrame(data, index=locations)

plt.figure(figsize=(12, 8))
sns.heatmap(df, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
plt.title('共享单车分布热力图', fontsize=14)
plt.xlabel('时间', fontsize=12)
plt.ylabel('停车点位', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bike_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# In[14]:


plt.figure(figsize=(14, 7))
selected = ['东门', '南门', '二食堂', '梅苑1栋']
for loc in selected:
    plt.plot(df.columns, df.loc[loc], 'o-', label=loc)

plt.title('重点区域单车数量时间趋势', fontsize=14)
plt.xlabel('时间', fontsize=12)
plt.ylabel('单车数量', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('time_trend.png', dpi=300, bbox_inches='tight')
plt.show()


# In[15]:


peak_hours = ['9:00', '12:00', '18:00']
peak_data = df[peak_hours]

plt.figure(figsize=(12, 6))
width = 0.25
x = np.arange(len(locations))

for i, hour in enumerate(peak_hours):
    plt.bar(x + i*width, peak_data[hour], width=width, label=hour)

plt.title('高峰期单车数量对比', fontsize=14)
plt.xlabel('停车点位', fontsize=12)
plt.ylabel('单车数量', fontsize=12)
plt.xticks(x + width, locations, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('peak_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# In[17]:


# 地理坐标
np.random.seed(42)
coords = {loc: (np.random.uniform(0,10), np.random.uniform(0,10)) for loc in locations}

plt.figure(figsize=(10, 8))
for loc in locations:
    x, y = coords[loc]
    size = df.loc[loc].mean() * 5  # 气泡大小反映平均存量
    plt.scatter(x, y, s=size, alpha=0.6, label=loc)
    plt.text(x, y+0.3, loc, ha='center', fontsize=8)

plt.title('单车分布气泡图（尺寸表示存量）', fontsize=14)
plt.xlabel('经度', fontsize=12)
plt.ylabel('纬度', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('geo_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# In[18]:


plt.figure(figsize=(12, 6))
df.T.boxplot()
plt.title('各点位单车数量分布箱线图', fontsize=14)
plt.xlabel('停车点位', fontsize=12)
plt.ylabel('单车数量', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




