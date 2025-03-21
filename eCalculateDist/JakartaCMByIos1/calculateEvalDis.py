import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm
from src.data_reader.dataset import RawDatasetSingapore
from src.model.model import CDCK2ForSingapore
import logging, time
import os
from datetime import datetime
import h5py

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

def evaluate_model(args, list_txt, logger):
    try:
        # Set device
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Load the model
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)

        # Remove 'module.' prefix if present
        state_dict = checkpoint['state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
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

def calculate_mean_distance(dist_matrix):
    """
    计算每条轨迹到所有轨迹的距离和，并取平均
    :param dist_matrix: 曼哈顿距离矩阵
    :return: 平均距离
    """
    # 计算每行的距离和，得到一个 (1, num_queries) 矩阵
    row_sums = torch.sum(dist_matrix, dim=1)
    # 取平均
    mean_distance = torch.mean(row_sums).item()
    return mean_distance

def process_features_and_evaluate(args, query_hdf5, database_hdf5, query_txt, database_txt):
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
    distances = calculate_manhattan_distance_gpu(query_features, database_features)
    
    # 计算平均距离
    average_distance = calculate_mean_distance(distances)
    logger.info(f"Average distance: {average_distance}")
    return average_distance

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

    # 摩托车或出租车
    motorcycle_hdf5 = 'eCalculateDist/JakartaCMByIos1/motorcycleAndIos.hdf5'
    motorcycle_txt = 'eCalculateDist/JakartaCMByIos1/motor500.txt'
    taxi_hdf5 = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/allJakartaByIos.hdf5'
    taxi_txt = 'eCalculateDist/JakartaCMByIos1/car500.txt'
    # 数据库中全是出租车
    database_hdf5 = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/allJakartaByIos.hdf5'
    database_txt = 'eCalculateDist/JakartaCMByIos1/databaseCar15000.txt'
    
    # 处理特征提取和距离计算
    average_distance_motor = process_features_and_evaluate(args, motorcycle_hdf5, database_hdf5, motorcycle_txt, database_txt)
    print(f"Average distance for motorcycle: {average_distance_motor}")
    average_distance_car = process_features_and_evaluate(args, taxi_hdf5, database_hdf5, taxi_txt, database_txt)
    print(f"Average distance for car: {average_distance_car}")
    average_difference = average_distance_motor - average_distance_car
    if average_difference is not None:
        print(f"Overall average distance: {average_difference}")
    else:
        print("Failed to calculate average distance.")

if __name__ == '__main__':
    main()
