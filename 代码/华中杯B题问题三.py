#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import folium  # 用于地理可视化

from matplotlib import rcParams

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['PingFang HK']  # 设置字体为PingFang HK

rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================
# 1. 数据预处理
# ======================

# 已从附件1加载数据到DataFrame
# 这里创建数据
data = {
    '点位名称': ['东门', '南门', '北门', '一食堂', '二食堂', '教学2楼', '梅苑1栋'],
    '7:00': [30, 20, 25, 100, 120, 50, 80],
    '9:00': [60, 10, 40, 30, 40, 180, 20],
    '12:00': [40, 50, 30, 150, 80, 30, 60],
    '经度': [116.35, 116.34, 116.36, 116.355, 116.353, 116.358, 116.351],  
    '纬度': [39.99, 39.98, 40.00, 39.995, 39.992, 39.998, 39.991]        
}
df = pd.DataFrame(data)

# ======================
# 2. 运营效率评价模型
# ======================

def evaluate_efficiency(df):
    """计算各点位效率指标"""
    # 计算利用率波动率（标准差/均值）
    time_columns = ['7:00', '9:00', '12:00']  # 实际应包含所有时间列
    df['利用率波动'] = df[time_columns].std(axis=1) / df[time_columns].mean(axis=1)
    
    # 计算高峰需求满足率（假设9:00为高峰）
    df['高峰满足率'] = df['9:00'] / df[time_columns].max(axis=1)
    
    # 综合评分（权重可调整）
    df['效率评分'] = 0.6*df['高峰满足率'] + 0.4*(1-df['利用率波动'])
    return df.sort_values('效率评分', ascending=False)

df_eval = evaluate_efficiency(df)
print("效率评价结果：")
print(df_eval[['点位名称', '效率评分']])

# ======================
# 3. 布局优化建议
# ======================

def optimize_layout(df, n_clusters=5):
    """基于聚类分析优化停车点位布局"""
    # 获取坐标数据
    coords = df[['经度', '纬度']].values
    
    # 寻找最优聚类数（轮廓系数法）
    silhouette_scores = []
    for k in range(2, min(10, len(coords))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(coords)
        silhouette_scores.append(silhouette_score(coords, labels))
    
    optimal_k = np.argmax(silhouette_scores) + 2  # +2因为range从2开始
    
    # 使用最优聚类数
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # 计算每个聚类的中心点（建议新增点位）
    centers = kmeans.cluster_centers_
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=df['cluster'], cmap='viridis', s=100)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
    plt.title('停车点位聚类优化建议')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.show()
    
    return centers

# 获取优化建议的中心点
new_locations = optimize_layout(df)
print("\n建议新增点位坐标：")
print(new_locations)

# ======================
# 4. 地理可视化（需实际坐标）
# ======================

def plot_geo_map(df, centers):
    """生成交互式地理地图"""
    campus_map = folium.Map(location=[df['纬度'].mean(), df['经度'].mean()], zoom_start=15)
    
    # 现有点位
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['纬度'], row['经度']],
            popup=f"{row['点位名称']} 效率:{row['效率评分']:.2f}",
            icon=folium.Icon(color='blue' if row['效率评分']>0.5 else 'gray')
        ).add_to(campus_map)
    
    # 建议新增点位
    for i, center in enumerate(centers):
        folium.Marker(
            location=[center[1], center[0]],
            popup=f"建议点位{i+1}",
            icon=folium.Icon(color='red')
        ).add_to(campus_map)
    
    return campus_map

# 生成地图（需实际经纬度数据）
# map_obj = plot_geo_map(df, new_locations)
# map_obj.save('optimized_map.html')


# In[ ]:




