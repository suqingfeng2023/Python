import pandas as pd
import numpy as np

# 1. 数据读取与清洗
def load_data():
    # 固定列名
    columns = ['日期', '时间', '东门', '南门', '北门', '一食堂', '二食堂', '三食堂',
               '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼', '计算机学院', '工程中心',
               '网球场', '体育馆', '校医院']

    df = pd.read_excel("附件1-共享单车分布统计表.xlsx", header=1, names=columns)

    df.replace("200+", 200, inplace=True)
    df.fillna(0, inplace=True)

    df['时间'] = pd.to_datetime(df['时间'].astype(str), errors='coerce').dt.time
    df['小时'] = df['时间'].apply(lambda x: x.hour if pd.notna(x) else np.nan)

    locations = columns[2:]

    return df, locations


# 2. 数据分析
def analyze(df, locations):
    print(f"\n识别到 {len(locations)} 个停车点位。")
    max_counts = df[locations].apply(pd.to_numeric, errors='coerce').max()
    total_bikes = max_counts.sum()
    print(f"✅ 校园共享单车总量（最大值法）：{int(total_bikes)} 辆")

    # 计算不同时间点的各停车点位数量分布
    hourly_avg = df.groupby('小时')[locations].mean()
    print("\n不同时间点各停车点位的数量分布：")
    print(hourly_avg)


# 3. 执行程序
if __name__ == '__main__':
    try:
        df, locations = load_data()
        analyze(df, locations)
        print("\n🎉 分析完成！")
    except Exception as e:
        print(f"\n❌ 出错了：{e}")
