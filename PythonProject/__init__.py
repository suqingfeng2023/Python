import os
import pandas as pd
from flask import Flask, render_template
from pyecharts import options as opts
from pyecharts.charts import Map, Line, Bar, Page

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
    region_stats = epidemic_data.groupby('地区名称')['现存确诊'].max().reset_index()
    region_data_pairs = list(zip(
        region_stats['地区名称'].tolist(),
        region_stats['现存确诊'].tolist()
    ))

    distribution_map = (
        Map()
        .add("现存确诊人数", region_data_pairs, maptype="香港特别行政区")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="香港各区现存确诊人数分布"),
            visualmap_opts=opts.VisualMapOpts(max_=region_stats['现存确诊'].max()),
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
        .add_yaxis("新增确诊", daily_summary['新增确诊'].tolist())
        .add_yaxis("新增康复", daily_summary['新增康复'].tolist())
        .add_yaxis("新增死亡", daily_summary['新增死亡'].tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="疫情每日新增趋势"),
            xaxis_opts=opts.AxisOpts(name="日期"),
            yaxis_opts=opts.AxisOpts(name="人数"),
            toolbox_opts=opts.ToolboxOpts(is_show=True))
    )
    return trend_chart


def generate_risk_level_barchart(epidemic_data):
    """生成风险等级分布柱状图"""
    risk_level_distribution = epidemic_data.groupby('风险等级')['地区名称'].nunique()

    risk_chart = (
        Bar()
        .add_xaxis(risk_level_distribution.index.tolist())
        .add_yaxis("地区数量", risk_level_distribution.tolist())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="不同风险等级的地区数量"),
            xaxis_opts=opts.AxisOpts(name="风险等级"),
            yaxis_opts=opts.AxisOpts(name="地区数量"),
            toolbox_opts=opts.ToolboxOpts(is_show=True))
    )
    return risk_chart


# 路由处理
@flask_app.route('/')
def show_dashboard():
    """主页面路由，展示疫情数据仪表盘"""
    processed_data = load_and_preprocess_data()

    visualization_page = Page()
    visualization_page.add(generate_region_distribution_map(processed_data))
    visualization_page.add(generate_daily_trend_chart(processed_data))
    visualization_page.add(generate_risk_level_barchart(processed_data))

    return render_template('dashboard.html',
                           chart=visualization_page.render_embed())


# 主程序入口
if __name__ == '__main__':
    flask_app.run(debug=True, port=DEFAULT_PORT)