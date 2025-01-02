import h5py
import numpy as np

def processed_hdf5_to_less_Feature(input_file, output_file):
    # 打开输入的 HDF5 文件以读取数据
    with h5py.File(input_file, 'r') as hdf_in:
        # 创建一个新的 HDF5 文件以保存处理后的数据
        with h5py.File(output_file, 'w') as hdf_out:
            # 遍历 HDF5 文件中的所有键
            for key in hdf_in.keys():
                # 获取对应轨迹的数据，假设每个轨迹是 [轨迹点数, 特征数]
                # 特征的顺序假设为：[经度, 纬度, 时间, 速度, 方位, 时间间隔, 空间间隔]
                data = hdf_in[key][:]
                
                # 限定轨迹点数范围为 200 到 500
                if 400 <= data.shape[0] <= 2000:
                    # 选择需要的前六个特征（去除时间间隔和空间间隔）
                    # 即 [经度, 纬度, 时间, 速度, 方位]
                    selected_data = data[:, :6]  # 提取前五列特征
                    
                    # 将处理后的数据保存到新的 HDF5 文件中
                    hdf_out.create_dataset(key, data=selected_data)
                    
                    print(f"Processed and saved data for {key} with shape {selected_data.shape}")
                else:
                    print(f"Skipped {key} due to point count {data.shape[0]}")

    print("All data has been processed and saved to the new file.")

def query_new_hdf5_file(processed_file):
    # 打开 HDF5 文件以读取数据
    with h5py.File(processed_file, 'r') as hdf:
        # 遍历 HDF5 文件中的所有键
        for key in hdf.keys():
            # 获取对应轨迹的数据
            data = hdf[key][:]
            
            # 输出轨迹的键和形状
            print(f"Key: {key}, Data shape: {data.shape}")
            
            # 显示前几个轨迹点的数据
            print(f"First 5 points of trajectory {key}:")
            print(data[:6, :])  # 输出前5个轨迹点的特征数据
            print("-" * 40)


if __name__ == '__main__':
    input_file = 'dataset/geoLife/geolife.hdf5'   # 替换为你的输入文件路径
    output_file = 'dataset/geoLife/1/processedGeolife.hdf5' # 替换为你要保存的新文件路径
    # 处理 HDF5 文件
    processed_hdf5_to_less_Feature(input_file, output_file)

    processed_file = 'dataset/geoLife/processedGeolife.hdf5'  # 替换为你要查看的文件路径
    # 查询新的 HDF5 文件
    # query_new_hdf5_file(processed_file)
