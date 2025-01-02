import os
import pandas as pd
import h5py
from tqdm import tqdm
"""用于验证在长数据集上训练，在未见过的短数据集上测试的效果"""
def process_plt_file(plt_file, max_points=1024):
    try:
        # 读取 plt 文件内容，跳过前 7 行
        data = pd.read_csv(plt_file, skiprows=7, header=None,
                           names=['trj_id', 'rawlat', 'rawlng', 'pingtimestamp', 'driving_mode', 'osname', 'speed', 'bearing', 'accuracy'])
        
        # 只保留所需的字段
        data = data[['rawlat', 'rawlng', 'pingtimestamp', 'speed', 'bearing']]
        
        # 确保所有列都是数值类型
        data = data.apply(pd.to_numeric, errors='coerce')

        # 如果轨迹点数少于阈值，则跳过此轨迹
        if len(data) < max_points:
            return None

        # 生成轨迹ID组合作为数据集名称
        trajectory_id = os.path.basename(plt_file).split('.')[0]
        dataset_name = f'Trajectory_{trajectory_id}'
        return dataset_name, data.to_numpy()
    except Exception as e:
        print(f"Error processing file {plt_file}: {e}")
        return None

def convert_plt_to_hdf5(plt_dir, hdf5_filename, max_points=500):
    plt_files = [os.path.join(plt_dir, file) for file in os.listdir(plt_dir) if file.endswith('.plt')]
    
    count = 0
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        # 使用 tqdm 显示进度条
        for plt_file in tqdm(plt_files, desc="Processing files", unit="file"):
            result = process_plt_file(plt_file, max_points)
            if result:
                count += 1
                dataset_name, data = result
                hdf5_file.create_dataset(dataset_name, data=data, compression="gzip")
    print(f"Total files processed: {count}")
def main():
    plt_dir = 'dataset/grab_possi/grab_possi_Singapore_ios/pltFiles'  # 替换为你的 plt 文件目录
    hdf5_filename = 'evalSimilary/singaporeSimilary/subsetForVal/singaporeForVal.hdf5'  # 替换为你想存储 hdf5 文件的路径
    convert_plt_to_hdf5(plt_dir, hdf5_filename, max_points=1024)
    print("All plt files have been converted to HDF5.")

if __name__ == "__main__":
    main()
