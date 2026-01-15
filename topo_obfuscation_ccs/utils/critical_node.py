import ast
import math
import numpy as np
from collections import defaultdict
from typing import Dict

def read_node_metrics(file_path):
    """
    读取节点指标数据文件，返回节点字典。
    """
    node_metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    # 尝试分割节点名称和指标字典
                    node, metrics_str = line.split(': ', 1)
                    metrics = ast.literal_eval(metrics_str)
                    node_metrics[node] = metrics
                except ValueError as e:
                    print(f"解析错误: {e}，问题行: {line}")
                    continue
    return node_metrics

def normalize_data(data):
    """
    对数据进行标准化处理。
    """
    normalized_data = defaultdict(list)
    for node, metrics in data.items():
        for metric, value in metrics.items():
            normalized_data[metric].append(value)
    
    min_values = {metric: min(values) for metric, values in normalized_data.items()}
    max_values = {metric: max(values) for metric, values in normalized_data.items()}
    
    normalized_metrics = {}
    for node, metrics in data.items():
        normalized_metrics[node] = {
            metric: (value - min_values[metric]) / (max_values[metric] - min_values[metric]) if max_values[metric] != min_values[metric] else 0
            for metric, value in metrics.items()
        }
    return normalized_metrics

def entropy_weight_method(data):
    """
    使用熵权法计算每个指标的权重。
    """
    metrics = list(next(iter(data.values())).keys())
    normalized_data = np.array([[data[node][metric] for metric in metrics] for node in data])
    
    # 检查每个指标的值是否全为零
    valid_metrics = [metric for metric in metrics if not np.allclose(normalized_data[:, metrics.index(metric)], 0)]
    
    if not valid_metrics:
        raise ValueError("所有指标的值全为零，无法计算权重。")
    
    # 仅使用有效的指标进行计算
    normalized_data = normalized_data[:, [metrics.index(metric) for metric in valid_metrics]]
    
    # 计算每个指标的熵值
    epsilon = 1e-12  # 避免除零
    p_ij = normalized_data / normalized_data.sum(axis=0)
    e_j = -np.sum((p_ij * np.log(p_ij + epsilon)), axis=0) / len(data)
    
    # 计算权重
    weights = (1 - e_j) / (1 - e_j).sum()
    return dict(zip(valid_metrics, weights))

def topsis_method(data, weights):
    """
    使用优劣解距离法（TOPSIS）计算每个节点的得分。
    """
    metrics = list(next(iter(data.values())).keys())
    normalized_data = np.array([[data[node][metric] for metric in metrics] for node in data])
    
    # 仅使用有效的指标进行计算
    valid_metrics = list(weights.keys())
    normalized_data = normalized_data[:, [metrics.index(metric) for metric in valid_metrics]]
    
    # 计算加权标准化决策矩阵
    weighted_data = normalized_data * np.array(list(weights.values()))
    
    # 确定理想解和负理想解
    ideal_best = weighted_data.max(axis=0)
    ideal_worst = weighted_data.min(axis=0)
    
    # 计算每个节点与理想解和负理想解的距离
    distance_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    
    # 计算得分
    scores = distance_worst / (distance_best + distance_worst)
    return dict(zip(data.keys(), scores))

def identify_key_nodes(file_path):
    """
    识别关键节点。
    """
    # 读取数据
    node_metrics = read_node_metrics(file_path)
    
    # 数据标准化
    normalized_metrics = normalize_data(node_metrics)
    
    # 计算权重
    weights = entropy_weight_method(normalized_metrics)
    
    # 计算得分
    scores = topsis_method(normalized_metrics, weights)
    
    # 返回TopK节点
    k = math.ceil(len(node_metrics) * 0.2)  # 节点总数的20%
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return key_nodes


def computer_nodes_score(file_path):
    """
    识别关键节点。
    """
    # 读取数据
    node_metrics = read_node_metrics(file_path)
    
    # 数据标准化
    normalized_metrics = normalize_data(node_metrics)
    
    # 计算权重
    weights = entropy_weight_method(normalized_metrics)
    
    # 计算得分
    scores = topsis_method(normalized_metrics, weights)
    
    # 返回TopK节点
    k = math.ceil(len(node_metrics) * 0.2)  # 节点总数的20%
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return key_nodes

def calculate_node_scores(file_path):
    """
    计算每个节点的得分，并返回所有节点及其得分的字典。
    """
    # 读取数据
    node_metrics = read_node_metrics(file_path)
    
    # 数据标准化
    normalized_metrics = normalize_data(node_metrics)
    
    # 计算权重
    weights = entropy_weight_method(normalized_metrics)
    
    # 计算得分
    scores = topsis_method(normalized_metrics, weights)
    
    # 返回所有节点及其得分
    return scores

def identify_key_nodes_from_dict(metrics_dict: Dict[str, Dict[str, float]]) -> list:
    """
    识别关键节点（基于变量，不依赖文件）
    """
    normalized_metrics = normalize_data(metrics_dict)
    weights = entropy_weight_method(normalized_metrics)
    scores = topsis_method(normalized_metrics, weights)
    
    # 选取得分最高的前 20% 作为关键节点
    k = math.ceil(len(metrics_dict) * 0.2)
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return key_nodes

def calculate_node_scores_from_dict(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    计算所有节点的得分（基于变量，不依赖文件）
    """
    normalized_metrics = normalize_data(metrics_dict)
    weights = entropy_weight_method(normalized_metrics)
    scores = topsis_method(normalized_metrics, weights)
    
    return scores
# def main():
#     file_path = 'metrics.txt'  # 数据文件路径
#     key_nodes = identify_key_nodes(file_path)
    
#     # 输出结果
#     print("识别的关键节点如下：")
#     for node, score in key_nodes:
#         print(f"节点 {node}: 得分 = {score:.4f}")

# if __name__ == "__main__":
#     main()