import os
import numpy as np
from tqdm import tqdm

def read_plt_file(file_path):
    # 跳过前7行，读取每个轨迹文件的数据
    with open(file_path, 'r') as f:
        lines = f.readlines()[7:]
    data = []
    for line in lines:
        parts = line.strip().split(',')
        data.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), parts[4], parts[5], float(parts[6]), int(parts[7]), float(parts[8])])
    return data

def write_plt_file(file_path, header, data):
    # 写入插值后的轨迹数据到新的plt文件中
    with open(file_path, 'w') as f:
        f.writelines(header)
        for row in data:
            f.write(','.join(map(str, row)) + '\n')

def interpolate_trajectory(data, N_target):
    N_current = len(data)

    # 只有当点数小于 N_target 时，才进行插值操作
    while N_current < N_target:
        new_data = []
        insert_even = True  # 控制插入位置：第一次插入在偶数点，第二次在奇数点，交替进行
        for i in range(len(data) - 1):
            P1 = data[i]
            P2 = data[i + 1]

            new_data.append(P1)  # 保留当前点

            # 检查 P2 和 P3 的 bearing（第八列）都不为 0
            if i < len(data) - 2 and data[i + 1][7] != 0 and data[i + 2][7] != 0:
                if insert_even:
                    # 插入第一个新点 x1：P1 和 P2 之间
                    lat_new = (P1[1] + P2[1]) / 2
                    lng_new = (P1[2] + P2[2]) / 2
                    timestamp_new = round((P1[3] + P2[3]) / 2, 1)
                    speed_new = (P1[6] + P2[6]) / 2
                    bearing_new = round((P1[7] + P2[7]) / 2)
                    accuracy_new = P1[8]

                    # 新插入点的其他属性与 P1 相同
                    P_new = [P1[0], lat_new, lng_new, timestamp_new, P1[4], P1[5], speed_new, bearing_new, accuracy_new]
                    new_data.append(P_new)

                insert_even = not insert_even  # 交替插入位置

        new_data.append(data[-1])  # 保留最后一个点
        data = new_data
        N_current = len(data)

    return data

def process_directory(input_dir, output_dir, N_target):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.plt')]
    
    # 使用 tqdm 显示处理进度
    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        with open(file_path, 'r') as f:
            header = f.readlines()[:7]

        data = read_plt_file(file_path)

        # 筛选：只处理点数在 300 到 2000 之间的轨迹
        if len(data) < 300 or len(data) > 2000:
            continue

        # 对轨迹进行插值
        interpolated_data = interpolate_trajectory(data, N_target)
        write_plt_file(output_path, header, interpolated_data)

# 使用示例
input_directory = 'dataset/grab_possi/grab_possi_Jakarta_ios/allIosPltFiles'
output_directory = 'dataset/grab_possi/grab_possi_Jakarta_ios/allIosPltFilesyEnhancement'
N_target = 1024  # 目标轨迹点数

process_directory(input_directory, output_directory, N_target)
