import os
import logging,time
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean, directed_hausdorff
from pyproj import Transformer
from fastdtw import fastdtw

# ========= 基础函数 =========

def latlon_to_utm(coords):
    transformer = Transformer.from_crs("epsg:4326", "epsg:32650", always_xy=True)
    return np.array([transformer.transform(lon, lat) for lat, lon in coords])

def trajectory_length(coords_utm):
    if len(coords_utm) < 2:
        return 0
    return np.sum(np.linalg.norm(coords_utm[1:] - coords_utm[:-1], axis=1))

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
    edr_dist = dp[n][m]
    norm = min(n, m) if min(n, m) > 0 else 1
    return edr_dist, edr_dist / norm

def lcss(P, Q, epsilon=10):
    n, m = len(P), len(Q)
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if euclidean(P[i - 1], Q[j - 1]) <= epsilon:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcss_len = dp[n][m]
    norm = min(n, m) if min(n, m) > 0 else 1
    return lcss_len, lcss_len / norm

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

# ========= 主处理函数 =========

def evaluate_hdf5_limited500(input_hdf5_path, enhanced_hdf5_path, key_txt_path):
    results = []
    selected_keys = []

    with h5py.File(input_hdf5_path, 'r') as orig_file, \
         h5py.File(enhanced_hdf5_path, 'r') as enh_file:

        all_keys = list(orig_file.keys())
        count = 0

        for key in tqdm(all_keys, desc="Selecting valid trajectories"):
            if key not in enh_file:
                continue

            traj_orig = orig_file[key][()]
            traj_enh = enh_file[key][()]
            if len(traj_orig) < 2 or len(traj_enh) < 2 or len(traj_orig) >= 900:
                continue

            # ✅ 满足条件
            selected_keys.append(key)
            count += 1
            if count == 500:
                break

        # 保存选中的键
        with open(key_txt_path, 'w') as f:
            for key in selected_keys:
                f.write(key + '\n')
        print(f"✅ 已选中并保存 {len(selected_keys)} 条轨迹键 到: {key_txt_path}")

        # === 正式评估这500条轨迹 ===
        for key in tqdm(selected_keys, desc="Evaluating selected 500"):
            traj_orig = orig_file[key][()]
            traj_enh = enh_file[key][()]
            coords_orig = traj_orig[:, :2]
            coords_enh = traj_enh[:, :2]

            utm_orig = latlon_to_utm(coords_orig)
            utm_enh = latlon_to_utm(coords_enh)
            orig_length = trajectory_length(utm_orig)
            if orig_length == 0:
                continue

            dtw_dist, _ = fastdtw(utm_orig, utm_enh, dist=euclidean)
            frechet_dist_val = frechet_distance(utm_orig, utm_enh)
            hausdorff_dist = max(
                directed_hausdorff(utm_orig, utm_enh)[0],
                directed_hausdorff(utm_enh, utm_orig)[0]
            )
            edr_dist, edr_norm = edr(utm_orig, utm_enh, epsilon=10)
            lcss_len, lcss_norm = lcss(utm_orig, utm_enh, epsilon=10)

            dtw_norm = dtw_dist / orig_length
            frechet_norm = frechet_dist_val / orig_length
            hausdorff_norm = hausdorff_dist / orig_length

            results.append({
                'traj': key,
                'dtw': dtw_dist,
                'dtw_norm': dtw_norm,
                'frechet': frechet_dist_val,
                'frechet_norm': frechet_norm,
                'hausdorff': hausdorff_dist,
                'hausdorff_norm': hausdorff_norm,
                'edr': edr_dist,
                'edr_norm': edr_norm,
                'lcss': lcss_len,
                'lcss_norm': lcss_norm
            })

    return pd.DataFrame(results)

# ========= 程序入口 =========

if __name__ == '__main__':
    input_hdf5 = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5'
    output_hdf5 = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allCarSingapore_ULI_1024.hdf5'
    save_dir = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/meanAndStd/singapore/uli'

    os.makedirs(save_dir, exist_ok=True)
    
    key_txt_path = os.path.join(save_dir, 'valid_500_keys_under900.txt')
    result_xlsx_path = os.path.join(save_dir, 'trajectory_eval_SPLINE_valid500_less900.xlsx')
    result_log_path = os.path.join(save_dir, 'trajectory_eval_SPLINE_valid500_less900_stats.txt')

    start_time = time.time()
    # 处理特征提取和排名计算
    df_result = evaluate_hdf5_limited500(input_hdf5, output_hdf5, key_txt_path)
    df_result.to_excel(result_xlsx_path, index=False)
    print(f"📄 评估结果保存至：{result_xlsx_path}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"process_hdf5_from_txt 运行时间: {elapsed_time:.2f} 秒")

    # 输出均值标准差
    mean_vals = df_result.mean(numeric_only=True)
    std_vals = df_result.std(numeric_only=True)

    with open(result_log_path, 'w') as f:
        f.write("评估指标统计（均值 & 标准差）：\n")
        for col in mean_vals.index:
            f.write(f"{col}: mean = {mean_vals[col]:.4f}, std = {std_vals[col]:.4f}\n")

    print(f"📊 指标统计保存至：{result_log_path}")
