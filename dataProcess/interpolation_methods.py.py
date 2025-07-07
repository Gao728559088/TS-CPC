import h5py
import numpy as np
from math import atan2, degrees
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # 使用非交互式的后端，专用于保存图片
from scipy.interpolate import CubicSpline



# === 角度差值计算函数（处理方向角的跳跃问题）===
def angular_diff(a, b):
    """
    计算两个角度之间的最小差值（结果在 0-180° 范围内），
    解决角度从 179° 跳到 -179° 导致误差过大的问题。
    """
    diff = a - b
    return abs((diff + 180) % 360 - 180)

# === 插值函数 ===
def swli_linear_interpolation_full(traj, N_target=1024, theta_threshold=2, window_size=4, stride=3, max_passes=2):
    """
    使用基于滑动窗口的线性插值方法，对轨迹进行插值填充，直到长度达到 N_target。
    - traj: 原始轨迹，形状为 [N, 5]，包含纬度、经度、时间戳、速度、方向角。
    - theta_threshold: 插值的方向变化阈值（低于该阈值认为轨迹“平直”）。
    - window_size: 滑动窗口的长度。
    - stride: 每次滑动的步长。
    - max_passes: 最多插值轮数，防止死循环。
    """

    # 分离轨迹每一列，转换成列表（方便插入新点）
    lat = list(traj[:, 0])
    lon = list(traj[:, 1])
    timestamp = list(traj[:, 2])
    speed = list(traj[:, 3])
    bearing = list(traj[:, 4])

    inserted_count = 0  # 插入点的累计数量
    pass_count = 0      # 当前是第几轮插值尝试

    # 外部插值循环（最多进行 max_passes 轮）
    while len(lat) < N_target and pass_count < max_passes:
        i = 0
        inserted_this_pass = 0  # 本轮插入点数清零

        # 滑动窗口扫描轨迹
        while len(lat) < N_target and i + window_size <= len(lat):
            # 取出当前窗口的经纬度
            lat_win = lat[i:i+window_size]
            lon_win = lon[i:i+window_size]

            # 计算方向角（theta）序列
            directions = []
            for j in range(1, window_size):
                dy = lat_win[j] - lat_win[j - 1]
                dx = lon_win[j] - lon_win[j - 1]
                theta = degrees(atan2(dy, dx))  # 转为角度
                directions.append(theta)

            # 计算方向角的“平滑度”指标，即角度跳变的平均值
            delta_diff = [abs(angular_diff(directions[k], directions[k - 1])) for k in range(1, len(directions))]
            delta_avg = sum(delta_diff) / len(delta_diff)


            # 判断是否满足插值条件（方向变化较小）
            if delta_avg < theta_threshold:
                insert_pos = i + 2  # 插值位置设为中间点后（比如在点2和3之间）

                p1, p2 = insert_pos - 1, insert_pos

                # 对每个字段进行线性插值
                new_lat = round(0.5 * (lat[p1] + lat[p2]), 7)
                new_lon = round(0.5 * (lon[p1] + lon[p2]), 7)
                new_ts = int(round(0.5 * (timestamp[p1] + timestamp[p2])))
                new_speed = round(0.5 * (speed[p1] + speed[p2]), 6)
                new_bearing = int(round(0.5 * (bearing[p1] + bearing[p2])))

                # 插入新点
                lat.insert(insert_pos, new_lat)
                lon.insert(insert_pos, new_lon)
                timestamp.insert(insert_pos, new_ts)
                speed.insert(insert_pos, new_speed)
                bearing.insert(insert_pos, new_bearing)

                inserted_this_pass += 1
                inserted_count += 1
                i += stride  # 跳过 stride 步
            else:
                i += 1  # 不插值时只滑动一步

        pass_count += 1
        if inserted_this_pass == 0:
            break  # 如果本轮没有插入任何点，提前终止

    # 最终只保留前 N_target 个点
    lat = np.array(lat[:N_target])
    lon = np.array(lon[:N_target])
    timestamp = np.array(timestamp[:N_target])
    speed = np.array(speed[:N_target])
    bearing = np.array(bearing[:N_target])

    return np.stack([lat, lon, timestamp, speed, bearing], axis=1)  # 拼接回轨迹数组

def adasyn_interpolation_full(traj, N_target=1024, random_state=42):
    lat, lon, timestamp, speed, bearing = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4]
    X = np.stack([speed, bearing], axis=1)

    if len(X) < 6 or len(X) >= N_target:
        return traj

    # 伪造标签：前一半为类0，后一半为类1
    split_idx = len(X) // 2
    y = np.array([0] * split_idx + [1] * (len(X) - split_idx))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 目标插值数量：确保至少插入 1 个点
    n_insert = max(1, N_target - len(X))
    label_1_count = (y == 1).sum()
    target_label_1 = label_1_count + n_insert

    try:
        ada = ADASYN(
            sampling_strategy={1: target_label_1},
            random_state=random_state,
            n_neighbors=min(5, len(X) - 1)
        )
        X_res, y_res = ada.fit_resample(X_scaled, y)
        X_new = scaler.inverse_transform(X_res[len(X):])
    except Exception as e:
        print(f"[ADASYN 插值失败] 原因: {e}")
        return traj

    if len(X_new) == 0:
        return traj

    # 构造新空间点位置（用 speed/bearing 近邻反插）
    try:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_scaled)
        _, indices = nn.kneighbors(scaler.transform(X_new))
    except Exception as e:
        print(f"[邻居查找失败]: {e}")
        return traj

    inserted_points = []
    for feat, (i1, i2) in zip(X_new, indices):
        alpha = np.random.uniform(0.3, 0.7)
        new_point = [
            (1 - alpha) * lat[i1] + alpha * lat[i2],
            (1 - alpha) * lon[i1] + alpha * lon[i2],
            int((1 - alpha) * timestamp[i1] + alpha * timestamp[i2]),
            feat[0],  # speed
            feat[1]   # bearing
        ]
        inserted_points.append(new_point)

    final_traj = np.concatenate([traj, np.array(inserted_points)], axis=0)
    final_traj = final_traj[np.argsort(final_traj[:, 2])]
    return final_traj[:N_target]

def smote_interpolation_full(traj, N_target=1024, random_state=42):
    lat, lon, timestamp, speed, bearing = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4]
    X = np.stack([speed, bearing], axis=1)
    
    if len(X) < 4 or len(X) >= N_target:
        return traj

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ⛳ 分出伪标签
    split_idx = len(X) // 2
    y = np.array([0] * split_idx + [1] * (len(X) - split_idx))

    # 目标数量
    target_class = 1
    current_count = np.sum(y == target_class)
    desired_count = N_target - (len(X) - current_count)
    if desired_count <= current_count:
        return traj  # 不需要插值

    try:
        smote = SMOTE(
            sampling_strategy={target_class: desired_count},
            random_state=random_state,
            k_neighbors=min(3, current_count - 1)
        )
        X_res, y_res = smote.fit_resample(X_scaled, y)
        X_new = scaler.inverse_transform(X_res[len(X):])
    except Exception as e:
        print(f"[SMOTE 插值失败] 原因: {e}")
        return traj

    # 空间位置插值
    try:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_scaled)
        _, indices = nn.kneighbors(scaler.transform(X_new))
    except Exception as e:
        print(f"[SMOTE 邻居查找失败]: {e}")
        return traj

    inserted_points = []
    for feat, (i1, i2) in zip(X_new, indices):
        alpha = np.random.uniform(0.3, 0.7)
        new_point = [
            (1 - alpha) * lat[i1] + alpha * lat[i2],
            (1 - alpha) * lon[i1] + alpha * lon[i2],
            int((1 - alpha) * timestamp[i1] + alpha * timestamp[i2]),
            feat[0],
            feat[1]
        ]
        inserted_points.append(new_point)

    final_traj = np.concatenate([traj, np.array(inserted_points)], axis=0)
    final_traj = final_traj[np.argsort(final_traj[:, 2])]
    return final_traj[:N_target]

def uniform_linear_interpolation_full(traj, N_target=1024):
    """
    等距线性插值：在轨迹上等间隔地插入点，直到长度达到 N_target。
    - traj: 原始轨迹，形状为 [N, 5]，列分别为 lat, lon, timestamp, speed, bearing。
    - N_target: 插值后的目标轨迹长度。
    """
    if traj.shape[0] >= N_target:
        return traj[:N_target]

    # 轨迹的经纬度提取为复数形式用于线性插值
    lat = traj[:, 0]
    lon = traj[:, 1]
    ts = traj[:, 2]
    speed = traj[:, 3]
    bearing = traj[:, 4]

    # 计算累积距离作为“进度轴”
    coords = np.stack([lat, lon], axis=1)
    deltas = np.diff(coords, axis=0)
    dist = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dist)])  # 累积距离（从0开始）

    # 创建新的等间隔位置
    new_dists = np.linspace(0, cumdist[-1], N_target)

    # 插值函数
    lat_interp = np.interp(new_dists, cumdist, lat)
    lon_interp = np.interp(new_dists, cumdist, lon)
    ts_interp = np.interp(new_dists, cumdist, ts)
    speed_interp = np.interp(new_dists, cumdist, speed)
    bearing_interp = np.interp(new_dists, cumdist, bearing)

    # 组合回轨迹数组
    traj_interp = np.stack([lat_interp, lon_interp, ts_interp, speed_interp, bearing_interp], axis=1)

    return traj_interp

def curvature_aware_interpolation_full(
        traj,
        N_target=1024,
        curvature_power=4.0,         # ↑ 值越大，高曲率段越密集
        min_pts_per_seg=0,           # 直线段可为 0
        random_state=None):
    """
    Curvature-aware interpolation (CAI).
    Inserts more points on high-curvature segments, fewer on straight ones.

    Parameters
    ----------
    traj : ndarray [N,5]
        Original trajectory [lat, lon, ts, speed, bearing].
    N_target : int
        Desired length after interpolation.
    curvature_power : float
        Exponent for weighting curvature ( >1 使⾼曲率段更密集 ).
    min_pts_per_seg : int
        Minimum points inserted per segment (0 = 可不插).
    random_state : int | None
        For reproducibility of small random jitters (optional).

    Returns
    -------
    traj_interp : ndarray [N_target,5]
    """
    rng = np.random.default_rng(random_state)

    if traj.shape[0] >= N_target or traj.shape[0] < 3:
        return traj[:N_target]      # 已足够或太短则直接返回

    lat, lon = traj[:, 0], traj[:, 1]
    ts, spd, brg = traj[:, 2], traj[:, 3], traj[:, 4]

    # --- 1. 计算每个段的“曲率权重” ---------------------------
    # 向量 v_i = P_{i+1} - P_i  （用经纬度当平面坐标近似）
    vecs = np.diff(np.stack([lat, lon], axis=1), axis=0)  # shape [N-1, 2]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    v_norm = vecs / norms                                  # 单位向量

    # 转折角 θ_i = arccos(v_{i-1}·v_i) ，首末段曲率设0
    dot_cos = (v_norm[:-1] * v_norm[1:]).sum(axis=1).clip(-1, 1)
    angles = np.arccos(dot_cos)            # rad, shape N-2
    curvature = np.concatenate([[0], angles, [0]])  # 首尾设0

    # 权重 = (curvature^p) , 若全部 0 则退化为均匀插值
    weights = curvature ** curvature_power
    if np.allclose(weights.sum(), 0):
        weights = np.ones_like(weights)

    # --- 2. 计算各段需要插入的数量 ------------------------------
    N_insert = N_target - len(lat)
    # baseline 每段先分到 min_pts_per_seg
    base = np.full(len(weights), min_pts_per_seg, dtype=int)
    remaining = N_insert - base.sum()
    remaining = max(remaining, 0)

    # 按权重分配其余插值点
    if remaining > 0:
        probs = weights / weights.sum()
        alloc = np.floor(probs * remaining).astype(int)
        # 纠正四舍五入后的差值
        diff = remaining - alloc.sum()
        if diff > 0:
            extra_idx = rng.choice(len(alloc), diff, replace=False, p=probs)
            alloc[extra_idx] += 1
    else:
        alloc = np.zeros_like(weights, dtype=int)

    pts_per_seg = base + alloc        # 每段最终插入点数

    # --- 3. 在线性空间插入坐标 / 时间 / 速度 / 航向 --------------
    new_lat, new_lon, new_ts, new_spd, new_brg = [], [], [], [], []

    for i in range(len(lat) - 1):
        new_lat.append(lat[i])
        new_lon.append(lon[i])
        new_ts.append(ts[i])
        new_spd.append(spd[i])
        new_brg.append(brg[i])

        k = pts_per_seg[i]            # 该段要插多少
        if k > 0:
            frac = np.linspace(0, 1, k + 2)[1:-1]  # 去掉 0 和 1
            for f in frac:
                new_lat.append((1 - f) * lat[i] + f * lat[i + 1])
                new_lon.append((1 - f) * lon[i] + f * lon[i + 1])
                new_ts.append(int((1 - f) * ts[i] + f * ts[i + 1]))
                new_spd.append((1 - f) * spd[i] + f * spd[i + 1])
                new_brg.append((1 - f) * brg[i] + f * brg[i + 1])

    # 追加最后一个原始点
    new_lat.append(lat[-1])
    new_lon.append(lon[-1])
    new_ts.append(ts[-1])
    new_spd.append(spd[-1])
    new_brg.append(brg[-1])

    traj_interp = np.stack(
        [new_lat, new_lon, new_ts, new_spd, new_brg], axis=1
    )

    # 若偶尔点数多于 N_target，直接裁剪
    return traj_interp[:N_target]

def spline_interpolation_full(traj, N_target=1024):
    """
    使用三次样条插值对轨迹进行补全。
    - 假设轨迹为 [lat, lon, timestamp, speed, bearing]，其中经纬度用于插值。
    - 插值点均匀地分布在原始时间范围内。

    缺点：可能会出现轨迹偏离、速度和角度不真实的情况。
    优点：轨迹平滑，常用于可视化。

    参数：
    - traj: 原始轨迹，形状 [N, 5]
    - N_target: 插值后目标轨迹点数

    返回：
    - traj_interp: 插值后的轨迹，形状 [N_target, 5]
    """

    if traj.shape[0] >= N_target or traj.shape[0] < 4:
        return traj[:N_target]  # 不插值（太短或足够）

    lat, lon = traj[:, 0], traj[:, 1]
    ts = traj[:, 2].astype(np.int64)

    # 用原始时间戳作为插值变量
    try:
        lat_spline = CubicSpline(ts, lat)
        lon_spline = CubicSpline(ts, lon)
    except Exception:
        return traj[:N_target]

    # 构建等间距的时间戳（线性拉伸）
    ts_new = np.linspace(ts[0], ts[-1], N_target).astype(np.int64)
    lat_new = lat_spline(ts_new)
    lon_new = lon_spline(ts_new)

    # 插值速度与航向（可选：直接线性插值）
    speed = traj[:, 3]
    bearing = traj[:, 4]
    speed_spline = CubicSpline(ts, speed)
    bearing_spline = CubicSpline(ts, bearing)
    speed_new = speed_spline(ts_new)
    bearing_new = bearing_spline(ts_new)

    traj_interp = np.stack([lat_new, lon_new, ts_new, speed_new, bearing_new], axis=1)
    return traj_interp[:N_target]

# === 插值的主函数：批量处理 HDF5 中的所有轨迹，支持多种插值方法 ===
def interpolate_hdf5_trajectories(input_hdf5_path, output_hdf5_path, N_target=1024, method='swli', print_examples=3):
    """
    对 HDF5 中所有轨迹进行插值：
    - 长度小于 N_target 的轨迹执行插值；
    - 长度大于等于 N_target 的轨迹原样保存；
    支持 swli、adasyn、smote 插值方法。
    """
    method_map = {
        'swli': swli_linear_interpolation_full,
        'adasyn': adasyn_interpolation_full,
        'smote': smote_interpolation_full,
        'uli': uniform_linear_interpolation_full,  # 添加等距插值
        'cai' : curvature_aware_interpolation_full,
        'spline': spline_interpolation_full,  
    }

    if method not in method_map:
        raise ValueError(f"Unsupported interpolation method: {method}. Choose from 'swli', 'adasyn', 'smote'.")

    interpolator = method_map[method]

    with h5py.File(input_hdf5_path, 'r') as input_file, \
         h5py.File(output_hdf5_path, 'w') as output_file:

        total = 0
        interpolated = 0
        copied = 0

        for i, traj_name in enumerate(tqdm(input_file.keys(), desc=f"Processing ({method})")):
            traj = input_file[traj_name][()]
            total += 1

            if traj.shape[0] < N_target:
                # 插值补充
                traj_interp = interpolator(traj, N_target=N_target)
                output_file.create_dataset(traj_name, data=traj_interp, compression="gzip")
                interpolated += 1
                if i < print_examples:
                    print(f"\n📌 示例 {i+1}: {traj_name}（插值）")
                    print("➡️ 原始点数:", traj.shape[0])
                    print("➡️ 插值后点数:", traj_interp.shape[0])
                    print("原始轨迹前3个点:", traj[:3])
                    print("插值后轨迹前3个点:", traj_interp[:3])
            else:
                # 点数够，不插值
                output_file.create_dataset(traj_name, data=traj, compression="gzip")
                copied += 1
                if i < print_examples:
                    print(f"\n📌 示例 {i+1}: {traj_name}（未插值）")
                    print("➡️ 点数:", traj.shape[0])
                    print("轨迹前3个点:", traj[:3])

        print(f"\n✅ 共读取 {total} 条轨迹，插值 {interpolated} 条，直接复制 {copied} 条，保存至 {output_hdf5_path}。")



# === 可视化插值前后的轨迹图 ===都在一个图中，废弃
def save_interpolation_visualizations(input_hdf5_path, output_hdf5_path, output_dir, num_samples=100):
    """
    可视化插值前后的轨迹图，保存为图片文件。
    - input_hdf5_path: 原始轨迹 HDF5 路径
    - output_hdf5_path: 插值后轨迹 HDF5 路径
    - output_dir: 图片保存目录
    - num_samples: 可视化插值的轨迹数量上限（默认100）
    """

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_hdf5_path, 'r') as f_in, h5py.File(output_hdf5_path, 'r') as f_out:
        count = 0
        for key in f_in.keys():
            traj_in = f_in[key][()]
            traj_out = f_out[key][()]
            
            # 仅处理插值后点数多于原始轨迹的情况，也就是判断是否进行了插值
            if traj_in.shape[0] < traj_out.shape[0]:
                # 仅对插值过的轨迹进行绘图
                plt.figure(figsize=(8, 6))
                
                # 原始轨迹（蓝色）
                plt.plot(traj_in[:, 1], traj_in[:, 0], 'bo-', label='Original', markersize=2)
                
                # 插值后轨迹（红色）
                plt.plot(traj_out[:, 1], traj_out[:, 0], 'ro-', label='Interpolated', markersize=1)

                plt.title(f'Trajectory {key} (Original {traj_in.shape[0]} → Interpolated 1024)')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # 保存图片
                save_path = os.path.join(output_dir, f"{key}.png")
                plt.savefig(save_path, dpi=200)
                plt.close()

                count += 1
                if count >= num_samples:
                    break

    print(f"\n✅ 已保存 {count} 条插值前后轨迹图像至文件夹：{output_dir}")


# 插值前后的轨迹对比图：原始轨迹图（蓝色点和线）插值后轨迹图（原始点蓝色，新增点红色）
def save_interpolation_comparison(input_hdf5_path, output_hdf5_path, output_dir, method_name='SW-LI',
                                   num_samples=100, point_size=4, alpha=0.8, traj_key=None):
    """
    Compare original and interpolated trajectories by plotting:
    - Original trajectory (blue dots and line)
    - Interpolated trajectory (original points in blue, inserted points in red)

    Each plot includes detailed labels and is saved to output_dir in two subfolders.

    Parameters:
    - input_hdf5_path: path to original trajectories
    - output_hdf5_path: path to interpolated trajectories
    - output_dir: directory to save the output plots
    - num_samples: number of samples to visualize (ignored if traj_key is set)
    - point_size: marker size
    - alpha: transparency for scatter plots
    - traj_key: (optional) specific trajectory key to visualize only one trajectory
    """
    original_dir = os.path.join(output_dir, "original")
    interpolated_dir = os.path.join(output_dir, "interpolated")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(interpolated_dir, exist_ok=True)

    with h5py.File(input_hdf5_path, 'r') as f_in, h5py.File(output_hdf5_path, 'r') as f_out:
        if traj_key is not None:
            keys_to_process = [traj_key] if traj_key in f_in.keys() else []
        else:
            keys_to_process = list(f_in.keys())

        count = 0
        for key in keys_to_process:
            traj_in = f_in[key][()]
            traj_out = f_out[key][()]
            print(f"{key}: original {traj_in.shape[0]} vs interpolated {traj_out.shape[0]}")
            if traj_in.shape[0] < traj_out.shape[0]:
                # --- Plot 1: Original trajectory ---
                plt.figure(figsize=(8, 6))
                plt.scatter(traj_in[:, 1], traj_in[:, 0], c='blue', s=point_size,
                            label=f'Original Points: {traj_in.shape[0]}', alpha=alpha)
                plt.title(f'Original Trajectory (ID: {key})', fontsize=14)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(original_dir, f"{key}.png"), dpi=300)
                plt.close()

                # --- Plot 2: Interpolated trajectory with original vs inserted points ---
                coords_in_set = set(map(tuple, traj_in[:, :2].round(7)))
                coords_out = list(map(tuple, traj_out[:, :2].round(7)))
                is_inserted = [pt not in coords_in_set for pt in coords_out]
                coords_out = np.array(coords_out)

                plt.figure(figsize=(8, 6))

                # Original points in blue
                original_points = coords_out[~np.array(is_inserted)]
                if len(original_points) > 0:
                    plt.scatter(original_points[:, 1], original_points[:, 0],
                                c='blue', s=point_size,
                                label=f'Original: {traj_in.shape[0]} pts', alpha=alpha)

                # Inserted points in red
                inserted_points = coords_out[np.array(is_inserted)]
                if len(inserted_points) > 0:
                    plt.scatter(inserted_points[:, 1], inserted_points[:, 0],
                                c='red', s=point_size,
                                label=f'Inserted: {traj_out.shape[0] - traj_in.shape[0]} pts', alpha=alpha)

                plt.title(f'{method_name} Interpolation (ID: {key})', fontsize=14)
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(os.path.join(interpolated_dir, f"{key}.png"), dpi=300)
                plt.close()

                count += 1
                if traj_key is None and count >= num_samples:
                    break

    print(f"\n✅ Saved {count} trajectory comparison plots to: {output_dir}")



# 直接对比插值前后的轨迹长度（点数）分布，清晰展示 SW-LI 带来的长度补充效果：
def compare_trajectory_lengths(hdf5_path_before, hdf5_path_after, save_path=None):

    def get_lengths(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            return [f[k].shape[0] for k in f]

    lengths_before = get_lengths(hdf5_path_before)
    lengths_after = get_lengths(hdf5_path_after)

    plt.figure(figsize=(10, 6))
    plt.hist(lengths_before, bins=30, alpha=0.6, label='Before Interpolation', color='royalblue')
    plt.hist(lengths_after, bins=30, alpha=0.6, label='After Interpolation', color='orange')
    plt.xlabel('Trajectory Length')
    plt.ylabel('Count')
    plt.title('Trajectory Length Distribution Before and After SW-LI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ 图像保存至 {save_path}")
    else:
        plt.show()
    
    plt.close()


# === 程序入口 ===
if __name__ == '__main__':
    input_hdf5 = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allProcessedSingapore.hdf5'
    output_hdf5 = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/grab_possi/grab_possi_Singapore_all/allCarSingapore_ULI_1024.hdf5'
    # 对 HDF5 文件中的轨迹进行插值处理
    # interpolate_hdf5_trajectories(input_hdf5, output_hdf5, N_target=1024, method='spline', print_examples=3)

    # 可视化前100条插值轨迹，生成插值前后对比图，原始轨迹为蓝色，插值点为红色
    vis_dir = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/vis_uli1024/bykey'
    save_interpolation_comparison(input_hdf5, output_hdf5, vis_dir, num_samples=200,method_name='ULI',traj_key='Trajectory_10284')












    # # 保存前100条插值轨迹的可视化图像 
    # output_dir = '/home/ubuntu/Data/gch/CpcForTrajectory/dataset/dataEnhancement/vis_swli800_trajectories'
    # # 这个函数是将插值前后的轨迹都放在一块，这样可以直观的看到插值没有影响到原轨迹的形状。
    # save_interpolation_visualizations(input_hdf5, output_hdf5, output_dir, num_samples=100)





# === 插值的主函数：批量处理 HDF5 中的所有轨迹 ===
# def interpolate_hdf5_trajectories(input_hdf5_path, output_hdf5_path, N_target=1024, print_examples=3):
#     """
#     对HDF5中所有轨迹进行处理：
#     - 长度小于N_target的轨迹执行插值；
#     - 长度大于等于N_target的轨迹原样保存；
#     统一保存到输出文件。
#     """
#     with h5py.File(input_hdf5_path, 'r') as input_file, \
#          h5py.File(output_hdf5_path, 'w') as output_file:

#         total = 0
#         interpolated = 0
#         copied = 0

#         for i, traj_name in enumerate(tqdm(input_file.keys(), desc="Processing trajectories")):
#             traj = input_file[traj_name][()]
#             total += 1

#             if traj.shape[0] < N_target:
#                 # 轨迹点少，插值补充到目标长度
#                 traj_interp = sw_linear_interpolation_full(traj, N_target=N_target)
#                 output_file.create_dataset(traj_name, data=traj_interp, compression="gzip")
#                 interpolated += 1
#                 if i < print_examples:
#                     print(f"\n📌 示例 {i+1}: {traj_name}（插值）")
#                     print("➡️ 原始点数:", traj.shape[0])
#                     print("➡️ 插值后点数:", traj_interp.shape[0])
#                     print("原始轨迹前3个点:", traj[:3])
#                     print("插值后轨迹前3个点:", traj_interp[:3])
#             else:
#                 # 轨迹点够多，直接复制保存
#                 output_file.create_dataset(traj_name, data=traj, compression="gzip")
#                 copied += 1
#                 if i < print_examples:
#                     print(f"\n📌 示例 {i+1}: {traj_name}（未插值）")
#                     print("➡️ 点数:", traj.shape[0])
#                     print("轨迹前3个点:", traj[:3])

#         print(f"\n✅ 共读取 {total} 条轨迹，插值 {interpolated} 条，直接复制 {copied} 条，保存至 {output_hdf5_path}。")
