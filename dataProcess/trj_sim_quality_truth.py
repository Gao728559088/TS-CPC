import time
import h5py 
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff, euclidean
from fastdtw import fastdtw
from pyproj import Transformer
from tqdm import tqdm
import os

# 经纬度转UTM，返回二维numpy数组，单位米
def latlon_to_utm(coords):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
    # 注意传入顺序 lon, lat
    utm_coords = np.array([transformer.transform(lon, lat) for lat, lon in coords])
    return utm_coords

# 计算轨迹空间长度（单位米）
def trajectory_length(coords_utm):
    if len(coords_utm) < 2:
        return 0
    distances = np.linalg.norm(coords_utm[1:] - coords_utm[:-1], axis=1)
    return np.sum(distances)

# EDR 距离
def edr(P, Q, epsilon=10):
    n, m = len(P), len(Q)
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if euclidean(P[i - 1], Q[j - 1]) <= epsilon:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    norm = min(n, m) if min(n, m) > 0 else 1
    return dp[n][m], dp[n][m] / norm

# LCSS 匹配长度
def lcss(P, Q, epsilon=10):
    n, m = len(P), len(Q)
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if euclidean(P[i - 1], Q[j - 1]) <= epsilon:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    norm = min(n, m) if min(n, m) > 0 else 1
    return dp[n][m], dp[n][m] / norm

# Fréchet 距离
def frechet_distance(P, Q):
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return float('inf')
    ca = np.zeros((n, m))
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(P[0] - Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                np.linalg.norm(P[i] - Q[j])
            )
    return ca[-1, -1]

def process_hdf5_trajectories(input_hdf5_path, output_excel_path, output_key_txt_path, log_path, max_len=900, max_trajs=500):
    # 第一步：先筛选轨迹键，写TXT
    selected_keys = []
    with h5py.File(input_hdf5_path, 'r') as f:
        for key in tqdm(f.keys(), desc="筛选轨迹键"):
            traj = f[key][()]
            if traj.shape[0] >= max_len:
                continue
            if traj.shape[0] < 4:
                continue
            selected_keys.append(key)
            if len(selected_keys) >= max_trajs:
                break

    # 保存轨迹键名TXT
    with open(output_key_txt_path, 'w') as f:
        for k in selected_keys:
            f.write(k + '\n')
    print(f"轨迹键名保存到：{output_key_txt_path}")

    # 第二步：加载选中轨迹，计算指标
    results = []
    with h5py.File(input_hdf5_path, 'r') as f:
        for key in tqdm(selected_keys, desc="计算轨迹指标"):
            traj = f[key][()]
            coords = traj[:, :2]  # [lat, lon]

            # 奇偶分轨迹
            odd = coords[::2]
            even = coords[1::2]

            odd_utm = latlon_to_utm(odd)
            even_utm = latlon_to_utm(even)
            all_utm = latlon_to_utm(coords)

            total_len = trajectory_length(all_utm)
            if total_len == 0:
                continue

            dtw_dist, _ = fastdtw(odd_utm, even_utm, dist=euclidean)
            frechet_dist = frechet_distance(odd_utm, even_utm)
            hausdorff_dist = max(
                directed_hausdorff(odd_utm, even_utm)[0],
                directed_hausdorff(even_utm, odd_utm)[0]
            )
            edr_dist, edr_norm = edr(odd_utm, even_utm, epsilon=10)
            lcss_len, lcss_norm = lcss(odd_utm, even_utm, epsilon=10)

            dtw_norm = dtw_dist / total_len
            frechet_norm = frechet_dist / total_len
            hausdorff_norm = hausdorff_dist / total_len

            results.append({
                'traj_key': key,
                'dtw': dtw_dist,
                'dtw_norm': dtw_norm,
                'frechet': frechet_dist,
                'frechet_norm': frechet_norm,
                'hausdorff': hausdorff_dist,
                'hausdorff_norm': hausdorff_norm,
                'edr': edr_dist,
                'edr_norm': edr_norm,
                'lcss': lcss_len,
                'lcss_norm': lcss_norm
            })

    # 保存结果到Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"处理完毕，保存轨迹相似度结果到：{output_excel_path}")

        # 计算均值
    mean_values = df[['dtw', 'dtw_norm', 'frechet', 'frechet_norm','hausdorff', 'hausdorff_norm','edr', 'edr_norm', 'lcss', 'lcss_norm']].mean()

    # 计算标准差
    std_values = df[['dtw', 'dtw_norm', 'frechet', 'frechet_norm','hausdorff', 'hausdorff_norm','edr', 'edr_norm', 'lcss', 'lcss_norm']].std()

    # 写入日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("各指标原始值与归一化值的均值：\n\n")
        for metric, value in mean_values.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\n各指标原始值与归一化值的标准差：\n\n")
        for metric, value in std_values.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\n结果已保存：\nExcel表格：{output_excel}\n日志文件：{log_path}")


def compute_statistics_from_excel(result_excel_path, log_path):
    """
    从结果 Excel 文件中计算各指标的均值和标准差，并写入日志文件。
    """
    df_result = pd.read_excel(result_excel_path)

    metrics = [
        'dtw', 'dtw_norm',
        'frechet', 'frechet_norm',
        'hausdorff', 'hausdorff_norm',
        'edr', 'edr_norm',
        'lcss', 'lcss_norm'
    ]

    # 计算均值和标准差
    mean_values = df_result[metrics].mean()
    std_values = df_result[metrics].std()

    # 写入日志
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("各指标原始值与归一化值的均值：\n\n")
        for metric, value in mean_values.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\n各指标原始值与归一化值的标准差：\n\n")
        for metric, value in std_values.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"📊 均值与标准差已保存到日志文件：{log_path}")



if __name__ == "__main__":
    start_time = time.time()
    input_hdf5 = "/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5"
    output_excel = "/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/meanAndStd/trajectory_similarity_500.xlsx"
    output_key_txt = "/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/meanAndStd/selected_500_keys.txt"
    log_path = "/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/meanAndStd/trajectory_similarity_stats.txt"
    # process_hdf5_trajectories(input_hdf5, output_excel, output_key_txt,log_path)
    elapsed = time.time() - start_time
    print(f"\n⏱️ 运行总时间: {elapsed:.2f} 秒")

    compute_statistics_from_excel(output_excel, log_path)
