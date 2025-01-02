import h5py
import numpy as np
import random

# 从txt文件中读取轨迹键，然后从HDF5文件中读取对应的轨迹数据，存入到新的HDF5文件中和txt文件中
def process_hdf5_from_txt(input_file, key_txt, output_hdf5, output_txt, max_trajectories=1000):
    # 打开原始的 HDF5 文件和输出文件
    with h5py.File(input_file, 'r') as f_in, \
         h5py.File(output_hdf5, 'w') as f_out, \
         open(output_txt, 'w') as txt_out, \
         open(key_txt, 'r') as f_keys:

        # 从 txt 文件中读取所有轨迹键
        all_keys = f_keys.read().splitlines()

        # 随机抽取指定数量的轨迹键
        selected_keys = random.sample(all_keys, min(max_trajectories, len(all_keys)))

        # 轨迹计数器
        count = 0

        # 遍历选中的轨迹键并处理
        for key in selected_keys:
            if key in f_in:
                data = f_in[key][()]  # 获取轨迹数据

                # 将数据写入输出 HDF5 文件
                f_out.create_dataset(key, data=data)
                txt_out.write(key + '\n')  # 将键写入输出 TXT 文件

                # 增加计数器
                count += 1

    print(f"处理完成！共随机处理了 {count} 条轨迹，并写入到 HDF5 和 TXT 文件中。")

# 示例用法
input_hdf5 = 'dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5'  # 输入的 HDF5 文件
key_txt = 'dataGenerate/allSingaporeGenerate/allSingaporeTrain.txt'  # 包含轨迹键的 txt 文件
output_hdf5 = 'evalSimilary/mknnSimilary/database_data.hdf5'  # 输出的 HDF5 文件
output_txt = 'evalSimilary/mknnSimilary/database_data.txt'  # 输出的轨迹键 txt 文件

# 运行处理函数，最多随机抽取 1000 条轨迹
process_hdf5_from_txt(input_hdf5, key_txt, output_hdf5, output_txt, max_trajectories=10000)
