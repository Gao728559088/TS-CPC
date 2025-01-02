import argparse
import random
import numpy as np
import torch
from src.data_reader.dataset import RawDatasetSingapore
from src.model.model import CDCK2ForSingapore
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm
import logging
import os
from datetime import datetime
import h5py

# 设置日志记录
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

# 设置随机种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算曼哈顿距离
def calculate_manhattan_distance_gpu(query_features, database_features):
    num_queries = query_features.size(0)
    num_databases = database_features.size(0)
    dist_matrix = torch.zeros((num_queries, num_databases), device=query_features.device)

    for i in tqdm(range(num_queries), desc="Calculating distances", unit="query"):
        dist_matrix[i, :] = torch.sum(torch.abs(query_features[i] - database_features), dim=1)

    return dist_matrix

# k-NN查询函数
def knn_query(dist_matrix, k=20):
    dist_matrix_cpu = dist_matrix.cpu().numpy()
    nearest_indices = np.argsort(dist_matrix_cpu, axis=1)[:, :k]
    return nearest_indices

# 提取特征的函数（直接在 process_features 中调用 evaluate_model）
def evaluate_model(args, list_txt, logger):
    try:
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Load the model
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)

        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)  # Load the modified state dict
        model.eval()

        dataset = RawDatasetSingapore(args.raw_hdf5, list_txt, args.trajectory_window)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        features = []
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Extracting features", unit="batch"):
                batch_data = batch_data.float().to(device)
                output_features = model.encoder(batch_data)
                features.append(output_features.view(output_features.size(0), -1).cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        logger.info(f"Feature vectors shape: {features.shape}")
        
        return torch.tensor(features).to(device)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return None

# 计算 k-NN 查询
def process_features_and_evaluate(args, query_hdf5, database_hdf5, query_txt, database_txt, k=20):
    logger = setup_logger(args.log_dir)
    set_random_seed(42)

    # 提取查询集特征
    args.raw_hdf5 = query_hdf5
    query_features = evaluate_model(args, query_txt, logger)
    if query_features is None:
        logger.error("Failed to extract query features.")
        return None

    # 提取数据库集特征
    args.raw_hdf5 = database_hdf5
    database_features = evaluate_model(args, database_txt, logger)
    if database_features is None:
        logger.error("Failed to extract database features.")
        return None

    # 计算曼哈顿距离
    dist_matrix = calculate_manhattan_distance_gpu(query_features, database_features)

    # 执行 k-NN 查询
    knn_result = knn_query(dist_matrix, k=k)

    return knn_result

# 计算准确率，至少匹配15个
def calculate_accuracy(knn_original, knn_transformed, k=20, min_matches=20):
    match_count = 0
    num_queries = len(knn_original)

    for orig, trans in zip(knn_original, knn_transformed):
        # 计算匹配的轨迹数量
        matches = len(set(orig) & set(trans))
        if matches >= min_matches:  # 至少匹配15个
            match_count += 1

    accuracy = match_count / num_queries
    return accuracy

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and extract feature vectors')
    parser.add_argument('--model-path', type=str, default='snapshot/singapore/cdc-2024-11-14_22_36_07-model_best.pth', help='Path to the trained model file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--trajectory-window', type=int, default=1024, help='Window length to sample from each trajectory')
    parser.add_argument('--timestep', type=int, default=16, help='Timestep for the model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    parser.add_argument('--log-dir', type=str, default='evalSimilary/mknnSimilary/logs', help='Directory to save log files')
    args = parser.parse_args()

    # 设置 HDF5 文件路径
    query_hdf5 = 'evalSimilary/mknnSimilary/query_data.hdf5'  # 查询集数据
    query_txt = 'evalSimilary/mknnSimilary/query_data.txt'  # 查询集文件（包含查询轨迹的键）
    database_hdf5 = 'evalSimilary/mknnSimilary/database_data.hdf5'  # 原始数据库数据
    database_txt = 'evalSimilary/mknnSimilary/database_data.txt'  # 数据库文件（包含数据库轨迹的键）
    database_hdf5_transformed = 'evalSimilary/mknnSimilary/transformed_data_distort1.hdf5'  # 变换后的数据库数据
    database_txt_transformed = 'evalSimilary/mknnSimilary/database_data.txt'  # 变换后的数据文件（键保持一致）

    # 设置日志
    logger = setup_logger(args.log_dir)
    set_random_seed(42)

    # 处理特征提取和 k-NN 查询
    knn_original = process_features_and_evaluate(args, query_hdf5, database_hdf5, query_txt, database_txt, k=30)
    knn_transformed = process_features_and_evaluate(args, query_hdf5, database_hdf5_transformed, query_txt, database_txt_transformed, k=30)
    
    # 输出结果
    if knn_original is not None and knn_transformed is not None:
        accuracy = calculate_accuracy(knn_original, knn_transformed, k=30, min_matches=30)
        logger.info(f"Accuracy of k-NN matching (at least 15 matches): {accuracy:.4f}")
        print(f"Accuracy of k-NN matching (at least 15 matches): {accuracy:.4f}")
    else:
        logger.error("Failed to calculate k-NN results.")
        print("Failed to calculate k-NN results.")

if __name__ == '__main__':
    main()
