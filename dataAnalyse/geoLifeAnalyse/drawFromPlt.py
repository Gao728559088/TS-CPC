import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

"""在对应的数据文件中生成可视化的轨迹图，并返回轨迹的统计信息。"""

"""计算两点间的距离（以米为单位）"""
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000  # 地球平均半径，单位为米
    return c * r

"""获取单个PLT文件中的轨迹点数、持续时间、总的距离"""
def calculate_trajectory_stats(data):
    num_points = len(data)
    start_time = datetime.strptime(data.iloc[0]['Date'] + ' ' + data.iloc[0]['Time'], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['Date'] + ' ' + data.iloc[-1]['Time'], '%Y-%m-%d %H:%M:%S')
    duration = (end_time - start_time).total_seconds()
    total_distance = 0
    for i in range(1, num_points):
        total_distance += haversine(data.iloc[i - 1]['Longitude'], data.iloc[i - 1]['Latitude'],
                                    data.iloc[i]['Longitude'], data.iloc[i]['Latitude'])
    return num_points, duration, total_distance

"""绘制单个plt的轨迹点图"""
def process_plt_file(plt_file):
    data = pd.read_csv(plt_file, skiprows=6, header=None, names=['Latitude', 'Longitude', 'Zero', 'Altitude', 'Days', 'Date', 'Time'])
    num_points, duration, total_distance = calculate_trajectory_stats(data)

    # 绘制轨迹图
    plt.figure(figsize=(10,8))
    plt.plot(data['Longitude'], data['Latitude'], marker='o', markersize=2, linestyle='-')
    plt.title('GPS Trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    stats_text = f"Points: {num_points}\nDuration: {duration} seconds\nTotal Distance: {total_distance:.2f} meters"
    plt.text(0.5, 0.5, stats_text, transform=plt.gcf().transFigure)
    png_file = os.path.splitext(plt_file)[0] + '.png'
    plt.savefig(png_file)
    plt.close()

    return num_points, duration, total_distance

def summarize_stats(stats):
    stats_df = pd.DataFrame(stats, columns=['NumPoints', 'Duration', 'TotalDistance'])
    summary = stats_df.describe()
    summary.loc['median'] = stats_df.median()
    return stats_df, summary

"""绘制轨迹统计信息"""
def plot_stats(directory,stats_df, summary, stat_name):
    plt.figure(figsize=(10,6))
    plt.plot(stats_df.index, stats_df[stat_name], marker='o')
    plt.title(f'{stat_name} of Trajectories')
    plt.xlabel('Trajectory ID')
    plt.ylabel(stat_name)

    # 标记统计值
    for stat in ['max', 'min', 'mean', '50%']:
        value = summary.loc[stat, stat_name]
        plt.axhline(y=value, color='r', linestyle='--')
        plt.text(0, value, f'{stat}: {value:.2f}', color='r', va='bottom')

    plt.savefig(directory+f'{stat_name}_stats.png')
    plt.close()

"""将统计信息写入markdown文件"""
def write_summary_to_markdown(summary, filename):
    with open(filename, 'w') as file:
        # 写入表格头部
        headers = ["Statistic"] + list(summary.columns)
        file.write('| ' + ' | '.join(headers) + ' |\n')
        file.write('|' + '---|' * len(headers) + '\n')

        # 写入表格数据
        for index, row in summary.iterrows():
            row_str = [f"{val:.2f}" if isinstance(val, float) else val for val in row]
            file.write('| ' + ' | '.join([str(index)] + row_str) + ' |\n')


"""处理目录中的所有PLT文件"""
def process_plt_files(directory):
    plt_files = glob.glob(os.path.join(directory, '*.plt'))
    num_files = len(plt_files)
    print(f"Found {num_files} .plt files in the directory.")

    stats = []
    for index, plt_file in enumerate(plt_files, start=1):
        stat = process_plt_file(plt_file)
        stats.append(stat)
        print(f"Processed {index}/{num_files} files. {'█' * index}{'.' * (num_files - index)}")

    print("All files have been processed.")

    stats_df, summary = summarize_stats(stats)
    for stat_name in ['NumPoints', 'Duration', 'TotalDistance']:
        plot_stats(directory, stats_df, summary, stat_name)

    write_summary_to_markdown(summary, directory+'trajectory_stats_summary.md')
    print("Summary statistics written to 'trajectory_stats_summary.md'.")

def main():
    directory = 'trajectory/Data/000/Trajectory/'  # 替换为你的.plt文件所在目录
    process_plt_files(directory)

if __name__ == "__main__":
    main()
