import os
import pandas as pd
from flask import Flask, render_template
from pyecharts import options as opts
from pyecharts.charts import Map, Line, Bar, Pie, Page

# 常量定义
DATA_FILE_PATH = r'D:\PythonCode\PythonProject\香港各区疫情数据_20250322.xlsx'
DEFAULT_PORT = 5001

# Flask 应用初始化
flask_app = Flask(__name__)


# 数据加载与预处理
def load_and_preprocess_data():
    """加载并预处理疫情数据"""
    epidemic_data = pd.read_excel(DATA_FILE_PATH)
    epidemic_data['报告日期'] = pd.to_datetime(epidemic_data['报告日期'])
    return epidemic_data


# 可视化图表生成函数
def generate_region_distribution_map(epidemic_data):
    """生成香港地区疫情分布地图"""
    # 香港标准地区名称映射
    hk_standard_names = {
        '中西区': '中西区',
        '东区': '东区',
        '南区': '南区',
        '湾仔': '湾仔区',
        '湾仔区': '湾仔区',
        '九龙城': '九龙城区',
        '九龙城区': '九龙城区',
        '观塘': '观塘区',
        '观塘区': '观塘区',
        '深水埗': '深水埗区',
        '深水埗区': '深水埗区',
        '黄大仙': '黄大仙区',
        '黄大仙区': '黄大仙区',
        '油尖旺': '油尖旺区',
        '油尖旺区': '油尖旺区',
        '离岛': '离岛区',
        '离岛区': '离岛区',
        '葵青': '葵青区',
        '葵青区': '葵青区',
        '北区': '北区',
        '西贡': '西贡区',
        '西贡区': '西贡区',
        '沙田': '沙田区',
        '沙田区': '沙田区',
        '大埔': '大埔区',
        '大埔区': '大埔区',
        '荃湾': '荃湾区',
        '荃湾区': '荃湾区',
        '屯门': '屯门区',
        '屯门区': '屯门区',
        '元朗': '元朗区',
        '元朗区': '元朗区'
    }

    region_stats = epidemic_data.groupby('地区名称')['现存确诊'].max().reset_index()

    # 标准化地区名称
    region_stats['地区名称'] = region_stats['地区名称'].map(hk_standard_names)
    region_stats = region_stats.dropna()  # 移除无法识别的地区

    region_data_pairs = list(zip(
        region_stats['地区名称'].tolist(),
        region_stats['现存确诊'].tolist()
    ))

    distribution_map = (
        Map()
        .add("现存确诊人数",
             region_data_pairs,
             "香港",  # 使用"香港"而不是"香港特别行政区"
             is_map_symbol_show=False)  # 不显示标记点
        .set_global_opts(
            title_opts=opts.TitleOpts(title="香港各区现存确诊人数分布"),
            visualmap_opts=opts.VisualMapOpts(
                max_=region_stats['现存确诊'].max(),
                is_piecewise=True,  # 分段显示
                pos_left="10%",
                pos_bottom="10%"
            ),
            toolbox_opts=opts.ToolboxOpts(is_show=True)
        )
    )
    return distribution_map


def generate_daily_trend_chart(epidemic_data):
    """生成每日疫情趋势折线图"""
    daily_summary = epidemic_data.groupby('报告日期')[
        ['新增确诊', '新增康复', '新增死亡']].sum()

    trend_chart = (
        Line()
        .add_xaxis(daily_summary.index.strftime('%Y-%m-%d').tolist())
        .add_yaxis("新增确诊", daily_summary['新增确诊'].tolist(),
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3))
        .add_yaxis("新增康复", daily_summary['新增康复'].tolist(),
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3))
        .add_yaxis("新增死亡", daily_summary['新增死亡'].tolist(),
                   is_smooth=True,
                   linestyle_opts=opts.LineStyleOpts(width=3))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="疫情每日新增趋势"),
            xaxis_opts=opts.AxisOpts(name="日期"),
            yaxis_opts=opts.AxisOpts(name="人数"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            datazoom_opts=[opts.DataZoomOpts()]  # 添加数据缩放
        )
    )
    return trend_chart


def generate_risk_level_barchart(epidemic_data):
    """生成风险等级分布柱状图"""
    risk_level_distribution = epidemic_data.groupby('风险等级')['地区名称'].nunique()

    risk_chart = (
        Bar()
        .add_xaxis(risk_level_distribution.index.tolist())
        .add_yaxis("地区数量", risk_level_distribution.tolist(),
                   category_gap="50%",  # 柱子间距
                   itemstyle_opts=opts.ItemStyleOpts(
                       opacity=0.8,
                       border_width=2
                   ))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="不同风险等级的地区数量"),
            xaxis_opts=opts.AxisOpts(name="风险等级"),
            yaxis_opts=opts.AxisOpts(name="地区数量"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            visualmap_opts=opts.VisualMapOpts(
                dimension=1,
                min_=0,
                max_=risk_level_distribution.max(),
                range_color=["#90EE90", "#FFA07A", "#FF4500"]
            )
        )
    )
    return risk_chart


def generate_region_pie_chart(epidemic_data):
    """生成各区现存确诊人数占比饼状图"""
    region_stats = epidemic_data.groupby('地区名称')['现存确诊'].max().reset_index()
    # 按现存确诊人数排序，取前10个地区，其余归为"其他地区"
    sorted_regions = region_stats.sort_values('现存确诊', ascending=False)
    top_regions = sorted_regions.head(10)
    other_regions = sorted_regions.iloc[10:]

    data_pairs = [
        (row['地区名称'], row['现存确诊'])
        for _, row in top_regions.iterrows()
    ]

    if not other_regions.empty:
        other_sum = other_regions['现存确诊'].sum()
        data_pairs.append(("其他地区", other_sum))

    pie_chart = (
        Pie()
        .add("",
             data_pairs,
             radius=["30%", "75%"],  # 环形饼图
             rosetype="radius")  # 南丁格尔玫瑰图
        .set_global_opts(
            title_opts=opts.TitleOpts(title="香港各区现存确诊人数占比"),
            legend_opts=opts.LegendOpts(
                orient="vertical",
                pos_top="15%",
                pos_left="2%"
            ),
            toolbox_opts=opts.ToolboxOpts(is_show=True)
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(
                formatter="{b}: {c} ({d}%)",
                color="#333"
            ),
        )
    )
    return pie_chart


# 路由处理
@flask_app.route('/')
def show_dashboard():
    """主页面路由，展示疫情数据仪表盘"""
    processed_data = load_and_preprocess_data()

    visualization_page = Page(layout=Page.SimplePageLayout)
    visualization_page.add(
        generate_region_distribution_map(processed_data),
        generate_daily_trend_chart(processed_data),
        generate_risk_level_barchart(processed_data),
        generate_region_pie_chart(processed_data)
    )

    return render_template('dashboard.html',
                           chart=visualization_page.render_embed())


# 主程序入口
if __name__ == '__main__':
    flask_app.run(debug=True, port=DEFAULT_PORT)