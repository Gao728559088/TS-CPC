import pandas as pd
import os

# Step 1: 读取合并后的 Parquet 文件
parquet_file = '/home/ubuntu/forDataSet/grab_possi/grab_possi/grab_possi_Jakarta.parquet'  # 替换为你的 Parquet 文件路径
df = pd.read_parquet(parquet_file)

# Step 2: 创建保存 plt 文件的目录
output_dir = '/home/ubuntu/forDataSet/grab_possi/grab_possi/grab_possi_Jakarta_all_new/ios/iosPltFiles'  # 替换为输出 plt 文件保存目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 3: 根据 trj_id 分组，并对每个组按 pingtimestamp 排序
grouped = df.groupby('trj_id')

# Step 4: 遍历每个 trj_id 组，按 pingtimestamp 排序，筛选 osname 为 ios 的轨迹，并导出为 plt 文件
for trj_id, group in grouped:
    # 筛选 osname 为 ios 且 driving_mode 为 car 的数据
    ios_group = group[(group['osname'] == 'ios')]
    if ios_group.empty:
        continue  # 如果没有 ios 的数据，则跳过该 trj_id
    
    # 按 pingtimestamp 排序
    group_sorted = ios_group.sort_values(by='pingtimestamp')

    # 提取各列数据
    driving_mode = group_sorted['driving_mode'].values
    osname = group_sorted['osname'].values
    pingtimestamp = group_sorted['pingtimestamp'].values
    rawlat = group_sorted['rawlat'].values
    rawlng = group_sorted['rawlng'].values
    speed = group_sorted['speed'].values
    bearing = group_sorted['bearing'].values
    accuracy = group_sorted['accuracy'].values

    # Step 5: 生成 plt 文件内容
    plt_content = f"Geolife Trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0\n{len(rawlat)}\n"
    plt_content += "trj_id, rawlat, rawlng, pingtimestamp, driving_mode, osname, speed, bearing, accuracy\n"  # 添加列标题

    # 将每个点的信息按顺序添加到 plt 文件中，每行以 trj_id 开头
    for lat, lng, timestamp, mode, os_name, spd, brg, acc in zip(rawlat, rawlng, pingtimestamp, driving_mode, osname, speed, bearing, accuracy):
        plt_content += f"{int(trj_id)},{lat},{lng},{timestamp},{mode},{os_name},{spd},{brg},{acc}\n"

    # Step 6: 保存每条轨迹为一个 plt 文件
    plt_file_path = os.path.join(output_dir, f'{int(trj_id)}.plt')
    with open(plt_file_path, 'w') as f:
        f.write(plt_content)

    print(f"Saved trajectory {trj_id} to {plt_file_path}")

print("所有轨迹文件已导出完毕。")
