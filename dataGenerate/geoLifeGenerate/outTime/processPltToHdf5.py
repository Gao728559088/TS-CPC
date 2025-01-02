import glob
import os
import h5py
import numpy as np
import pandas as pd
from geopy.distance import geodesic # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn # type: ignore
import io
"""加快的处理"""
# 计算每个轨迹点之间的速度
def calculate_speed(data):
    distances = calculate_space_interval(data)
    time_diffs = data['Days'].diff().fillna(0) * 24 * 3600  # 将天数转换为秒
    speed = distances / time_diffs  # 距离/时间 = 速度 (m/s)
    speed[time_diffs == 0] = 0  # 如果时间差为0，速度设为0
    return speed

# 计算每个点的空间间隔（两点间的距离，单位：米）
def calculate_space_interval(data):
    coords = data[['Latitude', 'Longitude']].values
    distances = [geodesic(coords[i - 1], coords[i]).meters for i in range(1, len(coords))]
    distances.insert(0, 0)  # 第一个点没有前置点，间隔为0
    return np.array(distances)

# 计算方位角（Bearing）
def calculate_bearing(data):
    coords = data[['Latitude', 'Longitude']].values
    bearings = [calculate_initial_compass_bearing(coords[i - 1], coords[i]) for i in range(1, len(coords))]
    bearings.insert(0, 0)  # 第一个点没有前置点，方位设为0
    return np.array(bearings)

# 计算两个地理坐标点之间的方位角
def calculate_initial_compass_bearing(pointA, pointB):
    import math
    lat1, lon1 = math.radians(pointA[0]), math.radians(pointA[1])
    lat2, lon2 = math.radians(pointB[0]), math.radians(pointB[1])

    diff_long = lon2 - lon1
    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))
    initial_bearing = math.atan2(x, y)

    # 将弧度转换为度数，并确保结果为0-360度
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

# 处理单个plt文件，返回包含7个字段的数据和对应的轨迹名
def process_plt_file(plt_file, trajectory_id, min_points=200, max_points=1024):
    try:
        # 读取文件并跳过前6行
        with open(plt_file, 'r') as f:
            content = f.readlines()

        data = pd.read_csv(io.StringIO(''.join(content[6:])), header=None,
                           names=['Latitude', 'Longitude', 'Zero', 'Altitude', 'Days', 'Date', 'Time'])

        data = data[['Latitude', 'Longitude', 'Altitude', 'Days']].apply(pd.to_numeric, errors='coerce').dropna()

        # 如果轨迹点数少于阈值，则跳过此轨迹
        if len(data) < min_points or len(data) > max_points:
            return None

        # 计算速度、方位、时间间隔和空间间隔
        data['Speed'] = calculate_speed(data)
        data['Bearing'] = calculate_bearing(data)

        # 移除第一个点，因为它没有前置点来计算间隔和方位
        data = data.iloc[1:]

        # 生成轨迹ID为 Trajectory_x 的键
        dataset_name = f'Trajectory_{trajectory_id}'
        return dataset_name, data.to_numpy()

    except Exception as e:
        print(f"Error processing file {plt_file}: {e}")
        return None

# 打印进度条
def print_progress(current, total):
    bar_length = 50
    progress = int(current / total * bar_length)
    bar = '#' * progress + '-' * (bar_length - progress)
    print(f'\r[{bar}] {current}/{total} files processed', end='', flush=True)

# 将所有plt文件转换为一个HDF5文件，按顺序存储键 Trajectory_x
def convert_all_to_hdf5(base_dir, hdf5_filename, min_points=100, max_points=2000):
    # 获取所有用户目录和 .plt 文件
    plt_files = glob.glob(os.path.join(base_dir, '*', 'Trajectory', '*.plt'))
    total_files = len(plt_files)
    processed_files = 0
    trajectory_id = 1

    # 创建线程池并处理文件
    data_to_write = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_plt_file, plt_file, trajectory_id, min_points, max_points) 
                   for trajectory_id, plt_file in enumerate(plt_files, start=1)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                dataset_name, data = result
                data_to_write.append((dataset_name, data))
            processed_files += 1
            print_progress(processed_files, total_files)

    # 将数据写入HDF5文件
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        for dataset_name, data in data_to_write:
            hdf5_file.create_dataset(dataset_name, data=data, compression="gzip")

# 示例调用
base_dir = '/home/ubuntu/forDataSet/geoLife/Data'
hdf5_filename = 'dataset/geoLife/geolife.hdf5'
convert_all_to_hdf5(base_dir, hdf5_filename)
