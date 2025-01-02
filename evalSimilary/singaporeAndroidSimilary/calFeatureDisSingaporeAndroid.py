import os
import argparse
import torch
from src.data_reader.dataset import RawDatasetSingapore
from src.model.model import CDCK2ForSingapore
from torch.utils.data import DataLoader # type: ignore
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime


def setup_logger(log_dir):
    # 检查日志目录是否存在：
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成唯一的日志文件名：
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建一个新的日志记录器对象，名称为当前模块的名称。
    logger = logging.getLogger(__name__)
    # 设置日志级别为 INFO，这意味着只有 INFO 级别及以上的日志消息才会被记录。
    logger.setLevel(logging.INFO)
    
    # 查日志记录器是否已经有处理器，以避免重复添加。
    if not logger.hasHandlers():
        # 创建一个文件处理器，用于将日志消息写入到指定的日志文件中。
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 创建一个控制台处理器，用于将日志消息输出到控制台。
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 建一个格式化器，定义日志消息的显示格式，包括时间、记录器名称、日志级别和消息内容。
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 将格式化器添加到文件处理器和控制台处理器中，确保日志消息按照指定格式显示。
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 将处理器添加到记录器
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

# 这段代码用于设置随机种子，以确保在每次运行时得到相同的随机数序列。
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 这段代码用于评估模型并提取特征向量
# 修改 evaluate_model 函数中的模型加载部分
def evaluate_model(args, logger):
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

        dataset = RawDatasetSingapore(args.raw_hdf5, args.eval_list, args.trajectory_window)
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

def calculate_manhattan_distance_gpu(feature_vectors):
    """
    使用GPU计算所有特征向量之间的曼哈顿距离
    :param feature_vectors: 特征向量张量
    :return: 曼哈顿距离矩阵
    """
    num_vectors = feature_vectors.size(0)
    dist_matrix = torch.zeros((num_vectors, num_vectors), device=feature_vectors.device)
    
    for i in tqdm(range(num_vectors), desc="Calculating distances", unit="vector"):
        # feature_vectors[i] - feature_vectors 计算第 i 个特征向量与所有特征向量的差值。
        dist_matrix[i, :] = torch.sum(torch.abs(feature_vectors[i] - feature_vectors), dim=1)
        # print(dist_matrix[i, :])

    return dist_matrix

"""convert_to_rank_matrix 函数将 dist_matrix 中每一行的距离转换为排名，生成一个新的排名矩阵 rank_matrix。
"""
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
    计算满足特定条件的排名值的平均值
    条件：index1 % 3 == 1 且 index2 == index1 + 1
    :param rank_matrix: 排名矩阵
    :return: 平均排名值
    在得到排名矩阵之后，我们通过计算平均排名值来度量轨迹之间的相似性
    我们取奇数点与原轨迹和以及其他轨迹进行对比。
    在理想的情况下，奇数点轨迹与自身的距离排名为1，与原轨迹的相似性排名为2，与偶数点的相似性为3，
    如果符合上述情况则认为有好的评价相似性的效果，排名加1，其他的话加偶数点的排名，然后取平均，
    越接近与1说明CPC提取的特征通过距离矩阵计算的相似性好。
    """
    num_vectors = rank_matrix.shape[0]
    ranks = []

    for i in range(num_vectors):
        if i % 3 == 1 and i + 1 < num_vectors:
            # # 0:原轨迹 1：奇数点轨迹 2：偶数点轨迹，1-1本身肯定rank为0，1-2，1-0都可能为次低（1）
            if (rank_matrix[i, i + 1].item() ==1):  
                ranks.append(1.0)
            else:
                ranks.append(rank_matrix[i, i + 1].item())
            print(rank_matrix[i, i - 1].item())
  
    if ranks:
        return np.mean(ranks)
    else:
        return None

def plot_distance_matrix(dist_matrix, output_dir, model_identifier, list_file_name):
    """
    绘制距离矩阵热力图
    :param dist_matrix: 曼哈顿距离矩阵
    :param output_dir: 保存热力图的文件夹路径
    :param model_identifier: 模型标识，用于在热力图上显示
    :param list_file_name: 评估列表文件名，用于生成唯一的文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dist_matrix = dist_matrix.cpu().numpy()  # 移动到CPU并转换为numpy数组
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap="viridis", cbar_kws={'label': 'Manhattan Distance'})
    plt.xlabel('Index 2')
    plt.ylabel('Index 1')
    title = 'Manhattan Distance Matrix'
    if model_identifier:
        title += f' - {model_identifier}'
    plt.title(title)
    
    output_file = os.path.join(output_dir, f'disMatrix_{list_file_name}.png')
    plt.savefig(output_file)
    plt.close()  # Close the plot to free memory

def process_feature_vectors(args, hdf5_file, list_file):
    logger = setup_logger(args.log_dir)
    
    # 设置随机种子
    set_random_seed(42)

    args.raw_hdf5 = hdf5_file
    args.eval_list = list_file

    features = evaluate_model(args, logger)
    if features is not None:
        logger.info("Feature vectors extracted successfully.")
        
        # 计算所有特征向量之间的曼哈顿距离
        dist_matrix = calculate_manhattan_distance_gpu(features)
        # logger.info(f"Distance matrix shape: {dist_matrix.size()}")
        # 将距离矩阵转换为排名矩阵
    
        rank_matrix = convert_to_rank_matrix(dist_matrix)
        # logger.info(f"Rank matrix shape: {rank_matrix.size()}")
        # 计算满足特定条件的排名值的平均值
    
        mean_rank = calculate_mean_rank(rank_matrix)
        if mean_rank is not None:
            logger.info(f"Mean rank for specified condition: {mean_rank}")
        else:
            logger.info("No elements satisfy the specified condition.")
        
        # 绘制距离矩阵热力图
        model_identifier = args.model_path.split('/')[-1].replace('-model_best.pth', '')
        list_file_name = os.path.basename(list_file).replace('.txt', '')
        plot_distance_matrix(dist_matrix, args.output_dir, model_identifier, list_file_name)
        
        return mean_rank
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and extract feature vectors')
    parser.add_argument('--model-path', type=str, default='snapshot/singaporeAndroid/cdc-2024-09-04_22_17_09-model_best.pth', help='Path to the trained model file')
    parser.add_argument('--raw-hdf5', type=str, default='evalSimilary/singaporeSimilary/subsetForVal/singaporeForVal.hdf5', help='Path to the HDF5 file containing raw data')
    # parser.add_argument('--eval-list', type=str, default='geoLifeDataGenerate/geoEvList.txt', help='Path to the file listing evaluation trajectories')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--trajectory-window', type=int, default=1024, help='Window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12, help='Timestep for the model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    parser.add_argument('--log-dir', type=str, default='evalSimilary/singaporeSimilary/logs', help='Directory to save log files')
    parser.add_argument('--output-dir', type=str, default='evalSimilary/singaporeSimilary/heatmaps', help='Directory to save heatmap images')
    args = parser.parse_args()
    
    logger = setup_logger(args.log_dir)
    
    files = [f for f in os.listdir('evalSimilary/singaporeAndroidSimilary/subsetForVal') if f.endswith('.hdf5')]
    mean_ranks = []

    for file in files:
        hdf5_file = os.path.join('evalSimilary/singaporeAndroidSimilary/subsetForVal', file)
        list_file = hdf5_file.replace('.hdf5', '.txt')
        
        if os.path.exists(list_file):
            mean_rank = process_feature_vectors(args, hdf5_file, list_file)
            list_file_name = os.path.basename(list_file).replace('.txt', '')
            logger.info(f"Mean rank of {hdf5_file} and {list_file}: {mean_rank}")
            if mean_rank is not None:
                mean_ranks.append(mean_rank)
    
    if mean_ranks:
        overall_mean_rank = np.mean(mean_ranks)
        logger.info(f"Overall mean rank: {overall_mean_rank}")
    else:
        logger.info("No valid mean ranks calculated.")

if __name__ == '__main__':
    main()
