import h5py
import numpy as np
import random

# 处理 HDF5 文件中的数据，将偶数点和奇数点分别写入到不同的 HDF5 文件，并将键写入到对应的 txt 文件
def process_hdf5_from_txt(input_file, key_txt, even_hdf5, odd_hdf5, even_txt, odd_txt, max_trajectories=1000):
    # 打开原始的HDF5文件
    with h5py.File(input_file, 'r') as f_in, \
         h5py.File(even_hdf5, 'w') as f_even, \
         h5py.File(odd_hdf5, 'w') as f_odd, \
         open(even_txt, 'w') as txt_even, \
         open(odd_txt, 'w') as txt_odd, \
         open(key_txt, 'r') as f_keys:

        # 从txt文件中读取所有轨迹键
        all_keys = f_keys.read().splitlines()

        # 随机抽取指定数量的轨迹键
        selected_keys = random.sample(all_keys, min(max_trajectories, len(all_keys)))

        # 轨迹计数器
        count = 0

        # 遍历选中的轨迹键并处理
        for key in selected_keys:
            if key in f_in:
                data = f_in[key][()]  # 获取轨迹数据

                # 分离偶数点和奇数点
                even_points = data[::2]  # 偶数点
                odd_points = data[1::2]  # 奇数点

                # 将偶数点写入 even HDF5 文件
                f_even.create_dataset(key, data=even_points)
                txt_even.write(key + '\n')  # 将键写入 even_keys.txt 文件

                # 将奇数点写入 odd HDF5 文件
                f_odd.create_dataset(key, data=odd_points)
                txt_odd.write(key + '\n')  # 将键写入 odd_keys.txt 文件

                # 增加计数器
                count += 1

    print(f"处理完成！共随机处理了 {count} 条轨迹，并写入到对应的 HDF5 和 TXT 文件中。")

# 示例用法
input_hdf5 = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/jakartaIosForClassifier.hdf5'  # 输入的HDF5文件
key_txt = 'dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassTest.txt'  # 包含轨迹键的txt文件
even_hdf5 = 'evalSimilary/jakartaIosForClassSimilary/database_data.hdf5'  # 偶数点HDF5文件
odd_hdf5 = 'evalSimilary/jakartaIosForClassSimilary/query_data.hdf5'  # 奇数点HDF5文件
even_txt = 'evalSimilary/jakartaIosForClassSimilary/database_data.txt'  # 偶数点的轨迹键txt文件
odd_txt = 'evalSimilary/jakartaIosForClassSimilary/query_data.txt'  # 奇数点的轨迹键txt文件

# 运行处理函数，最多随机抽取500条轨迹
process_hdf5_from_txt(input_hdf5, key_txt, even_hdf5, odd_hdf5, even_txt, odd_txt, max_trajectories=1000)
