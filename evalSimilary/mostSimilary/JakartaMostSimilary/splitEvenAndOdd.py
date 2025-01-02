import h5py
import numpy as np
import random
"""
我给你一个hdf5的数据集，你将其中的每条轨迹，将偶数点写入到一个hdf5以及将原轨迹的键写入到txt文件，
同时将奇数点写入到另一个hdf5以及将原轨迹的键写入到txt文件。
# 将偶数点放入到数据库，作为数据库数据
# 将奇数点作为查询数据
"""
def process_hdf5_random(input_file, even_hdf5, odd_hdf5, even_txt, odd_txt, num_samples=1000):
    # 打开原始的HDF5文件
    with h5py.File(input_file, 'r') as f_in, \
         h5py.File(even_hdf5, 'w') as f_even, \
         h5py.File(odd_hdf5, 'w') as f_odd, \
         open(even_txt, 'w') as txt_even, \
         open(odd_txt, 'w') as txt_odd:

        # 获取原始文件中的所有键
        all_keys = list(f_in.keys())

        # 随机选择 num_samples 个轨迹键
        selected_keys = random.sample(all_keys, min(num_samples, len(all_keys)))

        # 遍历选中的键
        for key in selected_keys:
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

    print(f"处理完成！随机选择了 {num_samples} 条轨迹进行划分，并写入到对应的HDF5和TXT文件中。")

# 示例用法
input_hdf5 = 'dataset/grab_possi/grab_possi_Jakarta_all/allAndroidJakarta.hdf5' # 将ios大于1024的轨迹数据集进行划分，也就是在训练ios过程中所用的数据集
even_hdf5 = 'evalSimilary/mostSimilary/JakartaMostSimilary/database_data.hdf5'  # 将偶数点放入到数据库，作为数据库数据
odd_hdf5 = 'evalSimilary/mostSimilary/JakartaMostSimilary/query_data.hdf5' # 将奇数点作为查询数据
even_txt = 'evalSimilary/mostSimilary/JakartaMostSimilary/database_data.txt'
odd_txt = 'evalSimilary/mostSimilary/JakartaMostSimilary/query_data.txt'

process_hdf5_random(input_hdf5, even_hdf5, odd_hdf5, even_txt, odd_txt, num_samples=1000)
