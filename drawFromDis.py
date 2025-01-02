import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def extract_data_from_log(log_file):
    """
    从日志文件中提取index1, index2和distance数据
    
    Args:
    log_file (str): 日志文件路径
    
    Returns:
    list: 包含(index1, index2, distance)元组的列表
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    data = []
    pattern = re.compile(r'index (\d+) and (\d+): ([\d.]+)')
    for line in lines:
        match = pattern.search(line)
        if match:
            index1, index2, distance = int(match.group(1)), int(match.group(2)), float(match.group(3))
            data.append((index1, index2, distance))
    
    return data

def create_distance_matrix(data):
    """
    创建距离矩阵
    
    Args:
    data (list): 包含(index1, index2, distance)元组的列表
    
    Returns:
    np.ndarray: 距离矩阵
    """
    max_index = max(max(index1, index2) for index1, index2, _ in data)
    matrix = np.zeros((max_index + 1, max_index + 1))
    
    for index1, index2, distance in data:
        matrix[index1, index2] = distance
    
    return matrix

def plot_distance_matrix(matrix, output_file=None):
    """
    绘制距离矩阵热力图
    
    Args:
    matrix (np.ndarray): 距离矩阵
    output_file (str, optional): 保存热力图的文件路径。如果为None，则显示热力图
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={'label': 'Manhattan Distance'})
    plt.xlabel('Index 2')
    plt.ylabel('Index 1')
    plt.title('Manhattan Distance Matrix')
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def process_log_and_plot(log_file, output_file=None):
    """
    处理日志文件并生成距离矩阵热力图
    
    Args:
    log_file (str): 日志文件路径
    output_file (str, optional): 保存热力图的文件路径。如果为None，则显示热力图
    """
    data = extract_data_from_log(log_file)
    distance_matrix = create_distance_matrix(data)
    plot_distance_matrix(distance_matrix, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process log file to create a distance matrix heatmap.')
    parser.add_argument('log_file', type=str, default='similary/calFeature.log', help='Path to the log file')
    parser.add_argument('--output_file', type=str, default='similary/dis_matrix.png', help='Path to save the output heatmap image (optional)')
    args = parser.parse_args()
    
    process_log_and_plot(args.log_file, args.output_file)

