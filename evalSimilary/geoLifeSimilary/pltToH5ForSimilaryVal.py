import os
import pandas as pd
import glob
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
    首先通过convert_all_to_hdf5(base_dir, hdf5_filename, min_points=1000)函数中的
    glob.glob(os.path.join(base_dir, '*'))获取所有用户目录，解释：这句话的结果是得到base_dir下的所有目录文件，也就是用户文件夹,如base_dir = 'Data'，则结果为['Data/User1', 'Data/User2', 'Data/User3']
    然后通过plt_files = [(os.path.basename(user_dir), file) for user_dir in user_dirs for file in glob.glob(os.path.join(user_dir, 'Trajectory', '*.plt'))]
    获得所有用户目录下的Trajectory文件夹下的plt文件，解释：这句话的结果是得到所有用户文件夹下的Trajectory文件夹下的plt文件，
    并以元组的形式存储，元组的第一个元素是用户文件夹的名字，第二个元素是plt文件的路径 如("User1", "Data/User1/Trajectory/file1.plt")

    第二步是通过线程池遍历的方式，循环调用process_plt_file(user_id, plt_file, min_points=1000)方法，得到用于存储在hDF5文件中的数据文件中名称和数据，即名称为User_1/Trajectory_12345

    然后全部写入到HDF5文件中，最后通过print_structure(hdf5_file)方法打印HDF5文件的结构并统计每个用户的轨迹数量

"""

"""处理单个plt文件，返回数据和对应的轨迹名，如果轨迹点数少于min_points，则返回None"""
def process_plt_file(user_id, plt_file, min_points=2000,max_points=7000):
    try:
        data = pd.read_csv(plt_file, skiprows=6, header=None,
                           names=['Latitude', 'Longitude', 'Zero', 'Altitude', 'Days', 'Date', 'Time'])
        # 移除Date和Time列
        data = data.drop(columns=['Date', 'Time'])
        # 确保所有列都是数值类型
        data = data.apply(pd.to_numeric, errors='coerce')

        # 如果轨迹点数少于阈值，则跳过此轨迹
        if len(data) < min_points or len(data)>max_points:
            return None

        trajectory_id = os.path.basename(plt_file).split('.')[0]
        # 生成用户ID和轨迹ID组合作为数据集名称 User_1/Trajectory_12345
        dataset_name = f'User_{user_id}/Trajectory_{trajectory_id}'
        return dataset_name, data.to_numpy()
    except Exception as e:
        print(f"Error processing file {plt_file}: {e}")
        return None

"""打印进度条"""
def print_progress(current, total):

    bar_length = 50
    progress = int(current / total * bar_length)
    bar = '#' * progress + '-' * (bar_length - progress)
    print(f'\r[{bar}] {current}/{total} files processed', end='', flush=True)


"""将所有plt文件转换为一个HDF5文件"""
def convert_all_to_hdf5(base_dir, hdf5_filename, min_points=2000,max_points=7000):
    # 获取所有用户目录和 .plt 文件
  
    user_dirs = glob.glob(os.path.join(base_dir, '*'))

    plt_files = [(os.path.basename(user_dir), file) for user_dir in user_dirs for file in glob.glob(os.path.join(user_dir, 'Trajectory', '*.plt'))]
    
    total_files = len(plt_files)
    processed_files = 0
    processed_and_kept_files = 0  # 新变量

    # 创建线程池并处理文件
    data_to_write = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_plt_file, user_id, plt_file, min_points,max_points) for user_id, plt_file in plt_files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                data_to_write.append(result)
                processed_and_kept_files += 1  # 更新保留文件数
            processed_files += 1
            print_progress(processed_files, total_files)  # 注意这里我们仍然基于总文件数来更新进度

    # 这里您可能想根据processed_and_kept_files来更新或打印一些信息

    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        for dataset_name, data in data_to_write:
            hdf5_file.create_dataset(dataset_name, data=data, compression="gzip")


"""
 递归地打印HDF5文件的结构并统计每个用户的轨迹数量
 :param hdf5_file: HDF5文件对象
"""
def print_structure(hdf5_file):

    user_trajectory_counts = {}  # 用于存储每个用户的轨迹数量

    # 在你的 convert_all_to_hdf5 函数中，虽然并没有显式地创建组，
    # 可以通过构造数据集名称来隐式地创建组的结构,
    # 如果你的数据集名称类似于 "User1/Trajectory_001"，那么在 HDF5 文件中会创建一个名为 "User1" 的组，
    # 并在其中创建一个名为 "Trajectory_001" 的数据集。
    def print_group(group, prefix=''):
        """
        打印HDF5组及其下的数据集，并统计轨迹数量
        :param group: HDF5组对象
        :param prefix: 当前层级的前缀，用于格式化输出
        """
        for key in group:
            if isinstance(group[key], h5py.Dataset):
                print(f"{prefix}/{key}: Dataset")
                user_key = prefix.split('/')[1]  # 假设用户ID总是路径的第二部分
                # 确保用户键存在于字典中
                if user_key not in user_trajectory_counts:
                    user_trajectory_counts[user_key] = 0
                user_trajectory_counts[user_key] += 1
            elif isinstance(group[key], h5py.Group):
                print(f"{prefix}/{key}: Group")
                print_group(group[key], prefix=prefix + '/' + key)

    print_group(hdf5_file)

    allcount=0
    # 打印每个用户的轨迹数量统计
    for user, count in user_trajectory_counts.items():
        allcount+=count
    print(f"Total trajectories: {allcount}")

    
def main():
    base_dir = '/home/ubuntu/forDataSet/geoLife/Data'  # 替换为你的数据目录
    hdf5_filename = 'evalSimilary/geoLifeSimilary/subsetForVal/geoLife.hdf5'  # 替换为你想存储hdf5文件的目录
    convert_all_to_hdf5(base_dir, hdf5_filename,min_points=200,max_points=700)  # 数据点数少于2000的轨迹将被丢弃
    print("\nConversion completed.")

    with h5py.File(hdf5_filename, 'r') as file:
        print_structure(file)


if __name__ == "__main__":
    main()
