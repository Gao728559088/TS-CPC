import h5py
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
import os

# # === 参数设置 ===
# hdf5_path = "/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allCarSingapore_SWLI_800.hdf5"  # ✅ 输入 HDF5 文件路径
# window_size = 4     # 滑动窗口的大小（即每次计算的点数）
# stride = 1          # 滑动步长
# theta_threshold = 5 # 红线标注的方向变化角度阈值，用于参考插值策略设置

# # === 函数：计算每个滑动窗口内的方向变化平均值 ===
# def collect_avg_theta_changes(traj, window_size=4, stride=1):
#     lat = traj[:, 0]  # 提取纬度
#     lon = traj[:, 1]  # 提取经度
#     avg_theta_changes = []  # 存储每个窗口的平均方向变化值

#     # 从轨迹的第一个点开始，依次滑动窗口
#     for i in range(0, len(lat) - window_size, stride):
#         directions = []  # 当前窗口内相邻点的方向角

#         # 计算窗口内相邻两个点的方向角
#         for j in range(1, window_size):
#             dx = lon[i + j] - lon[i + j - 1]  # 经度差
#             dy = lat[i + j] - lat[i + j - 1]  # 纬度差
#             theta = degrees(atan2(dy, dx))   # 使用 atan2 计算方向角（弧度转角度）
#             directions.append(theta)

#         # 计算相邻方向角之间的差异，取绝对值后求平均
#         delta = [abs(directions[k] - directions[k - 1]) for k in range(1, len(directions))]
#         avg_theta = sum(delta) / len(delta)
#         avg_theta_changes.append(avg_theta)

#     return avg_theta_changes  # 返回当前轨迹所有滑动窗口的平均方向变化

# # === 数据读取与处理 ===
# avg_changes_all = []  # 存储所有轨迹的方向变化值

# with h5py.File(hdf5_path, 'r') as f:
#     print(f"轨迹总数：{len(f)}")  # 输出总轨迹数
#     for i, k in enumerate(list(f.keys())[:100]):  # 仅分析前 100 条轨迹
#         traj = f[k][()]  # 读取一条轨迹数据为 NumPy 数组
#         if traj.shape[0] >= 10:  # 至少保证有足够点数才能滑窗分析
#             avg_changes = collect_avg_theta_changes(traj, window_size, stride)
#             avg_changes_all.extend(avg_changes)  # 汇总所有轨迹的平均方向变化

# # === 可视化方向变化的分布情况 ===
# plt.figure(figsize=(10, 6))  # 创建图形窗口，设定大小
# plt.hist(avg_changes_all, bins=100, alpha=0.7, color='skyblue', range=(0, 60))  # 绘制直方图，并限制横坐标范围为 0-60
# plt.axvline(theta_threshold, color='red', linestyle='--', label=f'theta_threshold = {theta_threshold}°')  # 添加红色阈值线
# plt.xlabel('Average ∆θ in sliding window (degrees)')  # 横坐标标签
# plt.ylabel('Count')  # 纵坐标标签
# plt.title('Distribution of Direction Changes in Trajectories')  # 图标题

# # 设置横坐标刻度，每 5 度一个刻度
# plt.xticks(range(0, 60, 5))

# plt.legend()       # 显示图例
# plt.grid(True)     # 显示网格
# plt.tight_layout() # 自动调整布局
# plt.show()         # 显示图像



# 插值前后的theta变化分布单独绘图
def plot_theta_change_distribution_single(hdf5_path, save_dir=None,
                                          theta_threshold=5, window_size=4, stride=1, num_trajectories=100):
    import h5py, os
    import numpy as np
    import matplotlib.pyplot as plt
    from math import atan2, degrees

    def collect_avg_theta_changes(traj):
        lat, lon = traj[:, 0], traj[:, 1]
        avg_theta_changes = []
        for i in range(0, len(lat) - window_size, stride):
            directions = []
            for j in range(1, window_size):
                dx = lon[i + j] - lon[i + j - 1]
                dy = lat[i + j] - lat[i + j - 1]
                theta = degrees(atan2(dy, dx))
                directions.append(theta)
            delta = [abs(directions[k] - directions[k - 1]) for k in range(1, len(directions))]
            avg_theta_changes.append(sum(delta) / len(delta))
        return avg_theta_changes

    avg_changes_all = []
    with h5py.File(hdf5_path, 'r') as f:
        for i, k in enumerate(list(f.keys())[:num_trajectories]):
            traj = f[k][()]
            if traj.shape[0] >= window_size:
                avg_changes_all.extend(collect_avg_theta_changes(traj))

    # 绘图
# 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(avg_changes_all, bins=100, alpha=0.7, color='skyblue', range=(0, 30))
    plt.axvline(theta_threshold, color='red', linestyle='--', label=f'Threshold = {theta_threshold}°')
    plt.xlabel('Average ∆θ in sliding window (degrees)')
    plt.ylabel('Count')
    plt.title('Distribution of Average Directional Change on Grab-Posisi Dataset')
    plt.xticks(range(0, 35, 5))  # 0到30，每5度一个刻度
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'direction_change_distribution_single.png')
        plt.savefig(filename)
        print(f"✅ 图像已保存至 {filename}")
    else:
        plt.show()

    plt.close()

# 插值前后的theta变化分布对比
def compare_theta_change_distributions(hdf5_path_1, hdf5_path_2,
                                       label_1='Dataset A', label_2='Dataset B',
                                       save_dir=None,
                                       theta_threshold=5, window_size=4, stride=1, num_trajectories=100):
    import h5py, os
    import numpy as np
    import matplotlib.pyplot as plt
    from math import atan2, degrees

    def collect_avg_theta_changes(hdf5_path):
        avg_all = []
        with h5py.File(hdf5_path, 'r') as f:
            for i, k in enumerate(list(f.keys())[:num_trajectories]):
                traj = f[k][()]
                if traj.shape[0] >= window_size:
                    lat, lon = traj[:, 0], traj[:, 1]
                    for j in range(0, len(lat) - window_size, stride):
                        directions = []
                        for d in range(1, window_size):
                            dx = lon[j + d] - lon[j + d - 1]
                            dy = lat[j + d] - lat[j + d - 1]
                            theta = degrees(atan2(dy, dx))
                            directions.append(theta)
                        delta = [abs(directions[t] - directions[t - 1]) for t in range(1, len(directions))]
                        avg_all.append(sum(delta) / len(delta))
        return avg_all

    changes_1 = collect_avg_theta_changes(hdf5_path_1)
    changes_2 = collect_avg_theta_changes(hdf5_path_2)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(changes_1, bins=100, alpha=0.5, label=label_1, range=(0, 60), color='royalblue')
    plt.hist(changes_2, bins=100, alpha=0.5, label=label_2, range=(0, 60), color='orange')
    plt.axvline(theta_threshold, color='red', linestyle='--', label=f'Threshold = {theta_threshold}°')
    plt.xlabel('Average ∆θ in sliding window (degrees)')
    plt.ylabel('Count')
    plt.title('Direction Change Comparison Between Datasets')
    plt.xticks(range(0, 65, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存或显示
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'direction_change_comparison.png')
        plt.savefig(filename)
        print(f"✅ 图像已保存至 {filename}")
    else:
        plt.show()

    plt.close()


def compute_avg_theta_per_traj(hdf5_path, window_size=4, stride=1, num_trajectories=2000):
    avg_thetas = []
    with h5py.File(hdf5_path, 'r') as f:
        for i, k in enumerate(list(f.keys())[:num_trajectories]):
            traj = f[k][()]
            if traj.shape[0] >= window_size:
                lat, lon = traj[:, 0], traj[:, 1]
                local_avgs = []
                for j in range(0, len(lat) - window_size, stride):
                    directions = []
                    for d in range(1, window_size):
                        dx = lon[j + d] - lon[j + d - 1]
                        dy = lat[j + d] - lat[j + d - 1]
                        theta = degrees(atan2(dy, dx))
                        directions.append(theta)
                    delta = [abs(directions[t] - directions[t - 1]) for t in range(1, len(directions))]
                    if delta:
                        local_avgs.append(sum(delta) / len(delta))
                if local_avgs:
                    avg_thetas.append(np.mean(local_avgs))
    return np.array(avg_thetas)


def compare_avg_theta_diff_plot(hdf5_path_before, hdf5_path_after, save_path=None):
    avg_before = compute_avg_theta_per_traj(hdf5_path_before)
    avg_after = compute_avg_theta_per_traj(hdf5_path_after)
    diffs = avg_after - avg_before

    # 可视化：直方图
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, color='seagreen', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')

    # 设置字体和坐标刻度
    plt.xlim(-4, 1)
    # plt.xticks([-4, -3, -2, -1, -0.5, 0, 1])  # 手动添加 -0.5，不改变坐标范围
    plt.xticks([-4, -3.5, -3,-2.5, -2,-1.5, -1, -0.5, 0, 0.5, 1])
    plt.title('Histogram of Avg Δθ Differences (After - Before)', fontsize=20)
    plt.xlabel('Difference in Avg Direction Change (degrees)', fontsize=16)
    plt.ylabel('Trajectory Count', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ 差值直方图已保存至 {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # 示例调用
    plot_theta_change_distribution_single(
        '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5',
        save_dir='/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/theta_thrshold/showVision',
        theta_threshold=5,
        window_size=4,
        stride=1,
        num_trajectories=100
    )

    # compare_theta_change_distributions(
    #     '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5',
    #     '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allCarSingapore_SWLI_800.hdf5',
    #     label_1='Original',
    #     label_2='Interpolated',
    #     save_dir='/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/theta_thrshold/showVision',
    #     theta_threshold=5,
    #     window_size=4,
    #     stride=1,
    #     num_trajectories=100
    # )
    # compare_avg_theta_diff_plot(
    #     '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5',
    #     '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allCarSingapore_SWLI_1024.hdf5',
    #     save_path='/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/theta_thrshold/showVision/avg_theta_diff_histogram.png'
    # )
