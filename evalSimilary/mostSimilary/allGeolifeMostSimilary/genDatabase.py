import h5py
"""这一步就是在生成最终的查询数据库
也就是在偶数点之后追加除了偶数点之外的所有轨迹
同时要保持原偶数点轨迹的顺序，所以要先将偶数点轨迹写入新的HDF5文件
"""
def create_new_hdf5_and_append_trajectories(even_hdf5, all_data_hdf5, even_txt, new_hdf5, new_txt):
    # 打开偶数点轨迹的 HDF5 文件和原始完整的 HDF5 文件
    with h5py.File(even_hdf5, 'r') as f_even, h5py.File(all_data_hdf5, 'r') as f_all, h5py.File(new_hdf5, 'w') as f_new:
        # 读取原偶数点轨迹的键
        with open(even_txt, 'r') as txt_file:
            even_keys = [line.strip() for line in txt_file.readlines()]  # 保持顺序

        # Step 1: 将偶数点HDF5文件中的数据按顺序复制到新的HDF5文件中
        for key in even_keys:
            data = f_even[key][()]  # 获取偶数点轨迹数据
            f_new.create_dataset(key, data=data)  # 写入新HDF5文件

        # Step 2: 新建一个空列表来存储新追加的轨迹键
        new_keys = []

        # Step 3: 遍历完整数据集中的所有轨迹键，将不在偶数点中的轨迹追加
        for key in f_all.keys():
            if key not in even_keys:  # 只追加偶数点HDF5文件中不存在的键
                data = f_all[key][()]  # 获取轨迹数据
                f_new.create_dataset(key, data=data)  # 将轨迹追加到新的HDF5文件中
                new_keys.append(key)  # 将新轨迹的键追加到新键列表

        # Step 4: 将原有的偶数点轨迹键和新追加的轨迹键写入新的txt文件
        with open(new_txt, 'w') as txt_file:
            # 写入偶数点轨迹键（保持顺序）
            for key in even_keys:
                txt_file.write(f"{key}\n")
            # 写入新追加的轨迹键
            for key in new_keys:
                txt_file.write(f"{key}\n")

        print(f"新的HDF5文件 {new_hdf5} 创建成功！新轨迹已添加，并且轨迹键已记录在 {new_txt} 中。")

# 示例用法
even_hdf5 = 'evalSimilary/mostSimilary/allGeolifeMostSimilary/database_data.hdf5'        # 偶数点HDF5文件
all_data_hdf5 = 'dataset/geoLife/1/processedGeolife.hdf5'  # 被追加的轨迹，也就是完整的HDF5文件
even_txt = 'evalSimilary/mostSimilary/allGeolifeMostSimilary/database_data.txt'      # 包含偶数点轨迹键的txt文件
new_hdf5 = 'evalSimilary/mostSimilary/allGeolifeMostSimilary/true_database_data.hdf5'  # 包含偶数点轨迹以及追加的轨迹
new_txt = 'evalSimilary/mostSimilary/allGeolifeMostSimilary/true_database_data.txt'  # 包含所有轨迹键的txt文件

create_new_hdf5_and_append_trajectories(even_hdf5, all_data_hdf5, even_txt, new_hdf5, new_txt)
