import h5py
import numpy as np
import random

class TrajectoryTransformer:
    def __init__(self, drop_rate=0, distort_rate=0.3, noise_factor=0.01):
        """
        初始化轨迹变换类
        :param drop_rate: 随机删除点的比例（百分比）
        :param distort_rate: 随机扭曲点的比例（百分比）
        :param noise_factor: 扭曲点时噪声的幅度
        """
        self.drop_rate = drop_rate
        self.distort_rate = distort_rate
        self.noise_factor = noise_factor

    def random_drop(self, trajectory):
        """
        随机删除轨迹中的点，确保删除点不连续
        :param trajectory: 原始轨迹
        :return: 删除点后的轨迹
        """
        n_points = len(trajectory)
        drop_count = int(n_points * self.drop_rate)
        
        if drop_count == 0:
            return trajectory

        # 确保删除的点均匀且不连续
        drop_indices = sorted(random.sample(range(0, n_points), drop_count))
        new_trajectory = [point for i, point in enumerate(trajectory) if i not in drop_indices]
        
        return new_trajectory

    # def random_distort(self, trajectory):
    #     """
    #     随机扭曲轨迹中的点，确保扭曲点不连续
    #     :param trajectory: 原始轨迹
    #     :return: 扭曲后的轨迹
    #     """
    #     n_points = len(trajectory)
    #     distort_count = int(n_points * self.distort_rate)
        
    #     if distort_count == 0:
    #         return trajectory

    #     # 扭曲的点均匀且不连续
    #     distort_indices = sorted(random.sample(range(0, n_points), distort_count))
        
    #     for i in distort_indices:
    #         lat, lng, timestamp, speed, bearing = trajectory[i]
    #         # 加入噪声来模拟扭曲
    #         lat += np.random.uniform(-self.noise_factor, self.noise_factor)
    #         lng += np.random.uniform(-self.noise_factor, self.noise_factor)
    #         trajectory[i] = (lat, lng, timestamp, speed, bearing)
        
    #     return trajectory
    def random_distort(self, trajectory):
        """
        随机扭曲轨迹中的点，采用 TrajRCL 的加噪方法
        :param trajectory: 原始轨迹
        :return: 扭曲后的轨迹
        """
        n_points = len(trajectory)
        distort_count = int(n_points * self.distort_rate)
        
        if distort_count == 0:
            return trajectory

        # 扭曲的点均匀且不连续
        distort_indices = sorted(random.sample(range(0, n_points), distort_count))
        
        for i in distort_indices:
            lat, lng, timestamp, speed, bearing = trajectory[i]
            # TrajRCL 方法: 加入 50m 范围内的高斯噪声
            lat += 50 * np.random.normal(0, 1)  # 高斯噪声
            lng += 50 * np.random.normal(0, 1)  # 高斯噪声
            trajectory[i] = (lat, lng, timestamp, speed, bearing)
        
        return trajectory

    def transform_with_drop(self, trajectory):
        """
        对单条轨迹进行删除点变换
        :param trajectory: 原始轨迹
        :return: 删除点后的轨迹
        """
        return self.random_drop(trajectory)

    def transform_with_distort(self, trajectory):
        """
        对单条轨迹进行扭曲点变换
        :param trajectory: 原始轨迹
        :return: 扭曲后的轨迹
        """
        return self.random_distort(trajectory)

def process_and_transform(input_hdf5, input_txt, output_hdf5_drop, output_hdf5_distort):
    # 初始化轨迹变换类
    transformer = TrajectoryTransformer(drop_rate=0.5, distort_rate=0.5, noise_factor=0.01)
    
    transformed_trajectories_drop = []  # 用于存储删除点后的轨迹
    transformed_trajectories_distort = []  # 用于存储扭曲点后的轨迹
    
    # 打开输入的 HDF5 文件和轨迹键的 txt 文件
    with h5py.File(input_hdf5, 'r') as f_in, open(input_txt, 'r') as txt_in:
        # 从 txt 文件中读取所有轨迹键
        keys = txt_in.read().splitlines()

        # 处理每一条轨迹
        for key in keys:
            if key in f_in:  # 如果轨迹键存在于HDF5文件中
                data = f_in[key][()]  # 获取轨迹数据
                trajectory = [(lat, lng, timestamp, speed, bearing) for lat, lng, timestamp, speed, bearing in data]  # 格式转换

                # # 对轨迹进行删除点变换
                # transformed_trajectory_drop = transformer.transform_with_drop(trajectory)
                # transformed_trajectories_drop.append((key, transformed_trajectory_drop))
                
                # 对轨迹进行扭曲点变换
                transformed_trajectory_distort = transformer.transform_with_distort(trajectory)
                transformed_trajectories_distort.append((key, transformed_trajectory_distort))
        
    # # 将变换后的轨迹保存到新的HDF5文件中（删除点后的轨迹）
    # with h5py.File(output_hdf5_drop, 'w') as f_out_drop:
    #     for key, traj in transformed_trajectories_drop:
    #         f_out_drop.create_dataset(key, data=np.array(traj))  # 保持键不变
    # print(f"删除点后的轨迹已保存到 {output_hdf5_drop}")

    # 将变换后的轨迹保存到新的HDF5文件中（扭曲点后的轨迹）
    with h5py.File(output_hdf5_distort, 'w') as f_out_distort:
        for key, traj in transformed_trajectories_distort:
            f_out_distort.create_dataset(key, data=np.array(traj))  # 保持键不变
    print(f"扭曲点后的轨迹已保存到 {output_hdf5_distort}")

# 示例用法
input_hdf5 = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data1.hdf5'  # 输入的 HDF5 文件
input_txt = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data.txt'  # 包含轨迹键的 txt 文件
output_hdf5_drop = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data.hdf5'  # 输出的删除点后轨迹的 HDF5 文件
output_hdf5_distort = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data.hdf5'  # 输出的扭曲点后轨迹的 HDF5 文件

# 运行处理和变换函数
process_and_transform(input_hdf5, input_txt, output_hdf5_drop, output_hdf5_distort)
