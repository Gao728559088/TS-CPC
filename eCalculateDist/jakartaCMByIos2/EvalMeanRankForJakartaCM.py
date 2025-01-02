import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm
from src.data_reader.dataset import  RawDatasetSingapore
from src.model.model import  CDCK2ForSingapore
import logging,time
import os
from datetime import datetime
import h5py
"""
这个文件主要是为了处理mostSimilary
就是先构建一个查询集，然后构建一个数据库集
对于查询集中的每一个轨迹，计算其与数据库集中所有轨迹的曼哈顿距离
然后对于每一个查询轨迹，计算其在所有数据库轨迹中的排名
最后计算平均排名
"""
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

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(args, list_txt,logger):
    try:
        # Set device
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Load the model
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)

        # Remove 'module.' prefix if present
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)  # Load the modified state dict
        model.eval()

        dataset = RawDatasetSingapore(args.raw_hdf5, list_txt,args.trajectory_window)
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
# 从 HDF5 文件中提取轨迹键
def load_keys_from_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        keys = list(f.keys())
    return keys

def calculate_manhattan_distance_gpu(query_features, database_features):
    """
    使用GPU计算查询集中的所有特征向量与数据库中所有特征向量之间的曼哈顿距离
    :param query_features: 查询集特征向量张量，形状 (num_queries, feature_dim)
    :param database_features: 数据库特征向量张量，形状 (num_databases, feature_dim)
    :return: 曼哈顿距离矩阵，形状 (num_queries, num_databases)
    """
    num_queries = query_features.size(0)
    num_databases = database_features.size(0)
    
    # 初始化距离矩阵
    dist_matrix = torch.zeros((num_queries, num_databases), device=query_features.device)
    
    # 计算每对查询特征与数据库特征之间的曼哈顿距离
    for i in tqdm(range(num_queries), desc="Calculating distances", unit="query"):
        dist_matrix[i, :] = torch.sum(torch.abs(query_features[i] - database_features), dim=1)

    return dist_matrix

def convert_to_rank_matrix(dist_matrix):
    """
    将距离矩阵转换为排名矩阵（每行单独排序）
    :param dist_matrix: 曼哈顿距离矩阵
    :return: 排名矩阵
    """
    rank_matrix = torch.zeros_like(dist_matrix, device=dist_matrix.device)
    
    for i in range(dist_matrix.size(0)):
        row = dist_matrix[i]
        rank_matrix[i] = row.argsort().argsort()  # 排序从0开始，默认去掉本身影响
    
    return rank_matrix

def calculate_mean_rank(rank_matrix):
    """
    """
    num_vectors = rank_matrix.shape[0]
    ranks = []

    for i in range(num_vectors):
            # # 0:原轨迹 1：奇数点轨迹 2：偶数点轨迹，1-1本身肯定rank为0，1-2，1-0都可能为次低（1）
            if (rank_matrix[i, i].item() == 0 ):  
                ranks.append(1.0)
            else:
                ranks.append(rank_matrix[i, i].item()+1)
                print(rank_matrix[i, i].item()+1)
    
    if ranks:
        return np.mean(ranks)
    else:
        return None

def process_features_and_evaluate(args, query_hdf5, database_hdf5,query_txt,database_txt):
    logger = setup_logger(args.log_dir)
    set_random_seed(42)

    # 提取查询集特征
    args.raw_hdf5 = query_hdf5
    query_features = evaluate_model(args, query_txt,logger)
    if query_features is None:
        logger.error("Failed to extract query features.")
        return None

    # 提取数据库集特征
    args.raw_hdf5 = database_hdf5
    database_features = evaluate_model(args, database_txt,logger)
    if database_features is None:
        logger.error("Failed to extract database features.")
        return None


    # 计算曼哈顿距离
    distances = calculate_manhattan_distance_gpu(query_features, database_features)
    
    # 获取排名矩阵
    rank_matrix = convert_to_rank_matrix(distances)
    
    # 计算平均排名
    average_rank = calculate_mean_rank(rank_matrix)
    logger.info(f"Average rank: {average_rank}")
    return average_rank

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and extract feature vectors')
    parser.add_argument('--model-path', type=str, default='snapshot/jakartaCMByIos/cdc-2024-10-31_11_23_10-model_best.pth', help='Path to the trained model file')
    parser.add_argument('--raw-hdf5', type=str, default='', help='其实就是一个中间参数')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--trajectory-window', type=int, default=1024, help='Window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12, help='Timestep for the model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    parser.add_argument('--log-dir', type=str, default='evalSimilary/mostSimilary/allJakartaByIosCMMostSimilary/logs', help='Directory to save log files')
    args = parser.parse_args()

    query_hdf5 = 'evalSimilary/mostSimilary/allJakartaByIosCMMostSimilary/query_data.hdf5'
    query_txt = 'evalSimilary/mostSimilary/allJakartaByIosCMMostSimilary/query_data.txt'
    database_hdf5 = 'evalSimilary/mostSimilary/allJakartaByIosCMMostSimilary/true_database_data.hdf5'
    database_txt = 'evalSimilary/mostSimilary/allJakartaByIosCMMostSimilary/true_database_data.txt'
    
    # 处理特征提取和排名计算
    average_rank = process_features_and_evaluate(args, query_hdf5, database_hdf5,query_txt,database_txt)
    if average_rank is not None:
        print(f"Overall average rank: {average_rank}")
    else:
        print("Failed to calculate average rank.")

if __name__ == '__main__':
 
    main()
   
