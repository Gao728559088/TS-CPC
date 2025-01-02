import os
import pandas as pd
import h5py
from tqdm import tqdm

def process_plt_file(plt_file, min_points=1024, max_points=2048, driving_mode_filter=None, osname_filter=None):
    try:
        # 读取 plt 文件内容，跳过前 7 行
        data = pd.read_csv(plt_file, skiprows=7, header=None,
                           names=['trj_id', 'rawlat', 'rawlng', 'pingtimestamp', 'driving_mode', 'osname', 'speed', 'bearing', 'accuracy'])
        
        # 将 driving_mode 列的字符串映射为数字
        data['driving_mode'] = data['driving_mode'].map({'car': 1, 'motorcycle': 0})
        driving_mode_value = data['driving_mode'].iloc[0]  # 获取第一个driving_mode值
        # 检查 driving_mode 是否符合指定条件，不符合则跳过
        if driving_mode_filter is not None:
            if not any(data['driving_mode'] == driving_mode_filter):
                return None  # 不包含指定交通模式，跳过该文件

        # 检查 osname 是否符合指定条件，不符合则跳过
        if osname_filter is not None:
            if not any(data['osname'] == osname_filter):
                return None  # 不包含指定操作系统，跳过该文件

        # 筛选出指定的交通模式和操作系统数据
        if driving_mode_filter is not None:
            data = data[data['driving_mode'] == driving_mode_filter]
        if osname_filter is not None:
            data = data[data['osname'] == osname_filter]

        # 如果筛选后数据为空或不符合点数条件，则跳过
        if data.empty or len(data) < min_points or len(data) > max_points:
            return None

        # 只保留所需的字段
        data = data[['rawlat', 'rawlng', 'pingtimestamp', 'speed', 'bearing']]
        
        # 确保所有列都是数值类型
        data = data.apply(pd.to_numeric, errors='coerce')

        # 生成轨迹ID组合作为数据集名称
        trajectory_id = os.path.basename(plt_file).split('.')[0]

        dataset_name = f'Trajectory_{trajectory_id}_{driving_mode_value}'
        
        return dataset_name, data.to_numpy()
    except Exception as e:
        print(f"Error processing file {plt_file}: {e}")
        return None
    
def convert_plt_to_hdf5(plt_dir, hdf5_filename, min_points=500, max_points=1024, driving_mode_filter=None, osname_filter=None):
    plt_files = [os.path.join(plt_dir, file) for file in os.listdir(plt_dir) if file.endswith('.plt')]
    
    count = 0
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # 使用 tqdm 显示进度条
        for plt_file in tqdm(plt_files, desc="Processing files", unit="file"):
            result = process_plt_file(plt_file, min_points, max_points, driving_mode_filter, osname_filter)
            if result:
                count += 1
                dataset_name, data = result
                hdf5_file.create_dataset(dataset_name, data=data, compression="gzip")
    print('经过处理后还剩多少条轨迹：', count)

def main():
    plt_dir = 'dataset/grab_possi/grab_possi_Jakarta_all/pltFiles'  # 替换为你的 plt 文件目录
    hdf5_filename = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/jakartaIosForClassifier.hdf5'  # 替换为你想存储 hdf5 文件的路径
    # 只处理 driving_mode 为 1（car）且 osname 为 'ios' 的数据
    convert_plt_to_hdf5(plt_dir, hdf5_filename, min_points=300, max_points=2000, driving_mode_filter=None, osname_filter='ios')
    print("All plt files have been converted to HDF5 with specified driving mode and osname.")

if __name__ == "__main__":
    main()
