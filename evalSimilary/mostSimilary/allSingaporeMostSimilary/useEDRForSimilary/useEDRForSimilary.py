import argparse
import numpy as np
import h5py
import logging
import os
from datetime import datetime
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

# 设置日志记录器
def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# 使用 GPU 加速的 LCSS 计算函数
def lcss_gpu(trajectory_a, trajectory_b, threshold, delta, device='cuda'):
    n = len(trajectory_a)
    m = len(trajectory_b)
    dp = torch.zeros((n + 1, m + 1), device=device)

    for i in range(1, n + 1):
        for j in range(max(1, i - delta), min(m + 1, i + delta)):
            dist = torch.norm(trajectory_a[i - 1] - trajectory_b[j - 1], p=2).item()
            if dist <= threshold:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    return dp[n, m].item()

# 加载 HDF5 数据中的轨迹
def load_trajectories_from_hdf5(hdf5_file, list_txt):
    with h5py.File(hdf5_file, 'r') as f:
        keys = []
        with open(list_txt, 'r') as txt_file:
            keys = [line.strip() for line in txt_file.readlines()]
        trajectories = [f[key][:] for key in keys]
    return trajectories

# 计算 LCSS 距离
def calculate_lcss_distance_pair(args):
    query_traj, database_traj, threshold, delta, device = args
    return lcss_gpu(query_traj, database_traj, threshold, delta, device)

def calculate_lcss_distances_gpu(query_trajectories, database_trajectories, threshold, delta, device='cuda'):
    num_queries = len(query_trajectories)
    num_databases = len(database_trajectories)
    dist_matrix = torch.zeros((num_queries, num_databases), device=device)

    # 将查询和数据库轨迹移到 GPU
    query_trajectories = [torch.tensor(traj, device=device) for traj in query_trajectories]
    database_trajectories = [torch.tensor(traj, device=device) for traj in database_trajectories]

    # 使用多线程计算 LCSS 距离
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in tqdm(range(num_queries), desc="Calculating LCSS distances", unit="query"):
            for j in range(num_databases):
                futures.append(executor.submit(calculate_lcss_distance_pair, (query_trajectories[i], database_trajectories[j], threshold, delta, device)))

        for index, future in enumerate(tqdm(futures, desc="Collecting results", unit="result")):
            i = index // num_databases
            j = index % num_databases
            dist_matrix[i, j] = future.result()

    return dist_matrix.cpu().numpy()  # 将结果转回 CPU

# 转换为排名矩阵
def convert_to_rank_matrix(dist_matrix):
    rank_matrix = np.zeros_like(dist_matrix)

    for i in range(dist_matrix.shape[0]):
        row = dist_matrix[i]
        rank_matrix[i] = row.argsort().argsort()

    return rank_matrix

# 计算平均排名
def calculate_mean_rank(rank_matrix):
    num_vectors = rank_matrix.shape[0]
    ranks = []

    for i in range(num_vectors):
        if rank_matrix[i, i] < 3:
            ranks.append(1.0)
        else:
            ranks.append(rank_matrix[i, i] + 1)
            print(rank_matrix[i, i] + 1)

    if ranks:
        return np.mean(ranks)
    else:
        return None

# 处理和评价
def process_and_evaluate(args, query_hdf5, database_hdf5, query_txt, database_txt, threshold=0.5, delta=2):
    logger = setup_logger(args.log_dir)

    # 加载查询集和数据库中的轨迹
    query_trajectories = load_trajectories_from_hdf5(query_hdf5, query_txt)
    database_trajectories = load_trajectories_from_hdf5(database_hdf5, database_txt)

    # 计算 LCSS 距离
    distances = calculate_lcss_distances_gpu(query_trajectories, database_trajectories, threshold, delta)

    # 获取排名矩阵
    rank_matrix = convert_to_rank_matrix(distances)

    # 计算平均排名
    average_rank = calculate_mean_rank(rank_matrix)
    logger.info(f"Average rank: {average_rank}")
    return average_rank

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory similarity using LCSS')
    parser.add_argument('--raw-hdf5', type=str, help='查询集和数据库的HDF5文件路径')
    parser.add_argument('--query-txt', type=str, help='查询集轨迹列表')
    parser.add_argument('--database-txt', type=str, help='数据库轨迹列表')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志文件保存路径')
    args = parser.parse_args()

    query_hdf5 = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/query_data.hdf5'
    query_txt = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/query_data.txt'
    database_hdf5 = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data.hdf5'
    database_txt = 'evalSimilary/mostSimilary/allSingaporeMostSimilary/true_database_data.txt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    average_rank = process_and_evaluate(args, query_hdf5, database_hdf5, query_txt, database_txt, threshold=0.5, delta=2)
    
    if average_rank is not None:
        print(f"Overall average rank: {average_rank}")
    else:
        print("Failed to calculate average rank.")

if __name__ == '__main__':
    main()
