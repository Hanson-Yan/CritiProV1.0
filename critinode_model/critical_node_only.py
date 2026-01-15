import ast
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# 定义需要使用的4个节点指标
TARGET_METRICS = [
    'betweenness_centrality',
    'closeness_centrality',
    'degree_centrality',
    'aggregate_traffic_load'
]

def read_node_metrics(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    读取节点指标数据文件，仅保留指定的4个指标
    :param file_path: 数据文件路径
    :return: 节点指标字典，格式为{节点名: {指标: 值}}
    """
    node_metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    # 分割节点名称和指标字典
                    node, metrics_str = line.split(': ', 1)
                    metrics = ast.literal_eval(metrics_str)
                    # 仅保留指定的4个指标
                    filtered_metrics = {m: metrics[m] for m in TARGET_METRICS if m in metrics}
                    node_metrics[node] = filtered_metrics
                except Exception as e:
                    print(f"解析错误: {e}，跳过行: {line}")
    return node_metrics

def normalize_data(data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    对节点数据进行标准化处理（最大-最小标准化）
    :param data: 节点指标字典
    :return: 标准化后的节点指标
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
            metric: (value - min_values[metric]) / (max_values[metric] - min_values[metric]) 
            if max_values[metric] != min_values[metric] else 0
            for metric, value in metrics.items()
        }
    return normalized_metrics

def entropy_weight_method(data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    使用熵权法计算指标权重
    :param data: 标准化后的节点指标
    :return: 指标权重字典
    """
    if not data:
        raise ValueError("输入数据不能为空")
    
    metrics = list(next(iter(data.values())).keys())
    normalized_array = np.array([[data[node][metric] for metric in metrics] for node in data])
    
    # 过滤全零指标
    valid_metrics = []
    valid_indices = []
    for i, metric in enumerate(metrics):
        if not np.allclose(normalized_array[:, i], 0):
            valid_metrics.append(metric)
            valid_indices.append(i)
    
    if not valid_metrics:
        raise ValueError("所有指标的值全为零，无法计算权重")
    
    # 使用有效指标计算
    valid_data = normalized_array[:, valid_indices]
    epsilon = 1e-12  # 避免除零
    
    # 计算概率矩阵
    p_ij = valid_data / valid_data.sum(axis=0)
    # 计算熵值
    e_j = -np.sum(p_ij * np.log(p_ij + epsilon), axis=0) / len(data)
    # 计算权重
    weights = (1 - e_j) / (1 - e_j).sum()
    
    return dict(zip(valid_metrics, weights))

def topsis_method(
    data: Dict[str, Dict[str, float]], 
    weights: Dict[str, float]
) -> Dict[str, float]:
    """
    使用TOPSIS法计算节点综合得分
    :param data: 标准化后的节点指标
    :param weights: 指标权重
    :return: 节点得分字典
    """
    if not data or not weights:
        raise ValueError("输入数据或权重不能为空")
    
    metrics = list(next(iter(data.values())).keys())
    normalized_array = np.array([[data[node][metric] for metric in metrics] for node in data])
    
    # 使用有权重的指标
    valid_metrics = list(weights.keys())
    valid_indices = [metrics.index(metric) for metric in valid_metrics]
    valid_data = normalized_array[:, valid_indices]
    
    # 计算加权标准化矩阵
    weighted_data = valid_data * np.array(list(weights.values()))
    
    # 确定理想解和负理想解
    ideal_best = weighted_data.max(axis=0)
    ideal_worst = weighted_data.min(axis=0)
    
    # 计算距离
    distance_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    
    # 计算得分
    scores = distance_worst / (distance_best + distance_worst)
    return dict(zip(data.keys(), scores))

def identify_key_nodes(
    file_path: str
) -> List[Tuple[str, float]]:
    """
    从文件中读取节点指标并识别关键节点（前20%）
    :param file_path: 节点指标文件路径
    :return: 关键节点列表，包含节点名和得分
    """
    # 读取数据并筛选指标
    node_metrics = read_node_metrics(file_path)
    if not node_metrics:
        return []
    
    # 数据标准化
    normalized_metrics = normalize_data(node_metrics)
    # 计算权重
    weights = entropy_weight_method(normalized_metrics)
    # 计算得分
    scores = topsis_method(normalized_metrics, weights)
    
    # 选取前20%的节点
    k = math.ceil(len(node_metrics) * 0.2)
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return key_nodes

def save_key_nodes_to_file(key_nodes: List[Tuple[str, float]], output_file: str):
    """将关键节点保存到文件"""
    with open(output_file, 'w') as f:
        # f.write(f"# 关键节点（前20%，共{len(key_nodes)}条）\n")
        for i, (node, score) in enumerate(key_nodes, 1):
            f.write(f"节点 {node}: 得分 = {score:.6f}\n")
    print(f"关键节点已保存至 {output_file}")

def main():
    # 输入输出文件路径
    input_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/metrics.txt"   # 包含节点指标的文件
    output_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/critical_node_only.txt"  # 关键节点输出文件
    
    try:
        # 识别关键节点
        key_nodes = identify_key_nodes(input_file)
        
        # 输出结果
        if key_nodes:
            print(f"识别到 {len(key_nodes)} 个关键节点（前20%）:")
            for i, (node, score) in enumerate(key_nodes, 1):
                print(f"{i}. 节点 {node}: 得分 = {score:.6f}")
            # 保存到文件
            save_key_nodes_to_file(key_nodes, output_file)
        else:
            print("未识别到关键节点")
            
    except Exception as e:
        print(f"执行过程中出错: {e}")

if __name__ == "__main__":
    main()