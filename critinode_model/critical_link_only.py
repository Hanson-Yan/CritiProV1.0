import ast
import math
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List

def read_link_metrics(file_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    读取链路指标数据文件，返回链路字典
    :param file_path: 数据文件路径
    :return: 链路指标字典，格式为{(s1,s2): {'ebc':..., 'cbr':...}}
    """
    link_metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    # 分割链路对和指标字典
                    link_str, metrics_str = line.split(': ', 1)
                    # 解析链路对（去除括号并分割节点）
                    link = ast.literal_eval(link_str)
                    metrics = ast.literal_eval(metrics_str)
                    link_metrics[link] = metrics
                except (ValueError, SyntaxError) as e:
                    print(f"解析错误: {e}，问题行: {line}")
                    continue
    return link_metrics

def normalize_data(data: Dict[Tuple[str, str], Dict[str, float]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    对链路数据进行标准化处理（最大-最小标准化）
    :param data: 链路指标字典
    :return: 标准化后的链路指标
    """
    normalized_data = defaultdict(list)
    for link, metrics in data.items():
        for metric, value in metrics.items():
            normalized_data[metric].append(value)
    
    min_values = {metric: min(values) for metric, values in normalized_data.items()}
    max_values = {metric: max(values) for metric, values in normalized_data.items()}
    
    normalized_metrics = {}
    for link, metrics in data.items():
        normalized_metrics[link] = {
            metric: (value - min_values[metric]) / (max_values[metric] - min_values[metric]) 
            if max_values[metric] != min_values[metric] else 0
            for metric, value in metrics.items()
        }
    return normalized_metrics

def entropy_weight_method(data: Dict[Tuple[str, str], Dict[str, float]]) -> Dict[str, float]:
    """
    使用熵权法计算指标权重
    :param data: 标准化后的链路指标
    :return: 指标权重字典
    """
    if not data:
        raise ValueError("输入数据不能为空")
    
    metrics = list(next(iter(data.values())).keys())
    normalized_array = np.array([[data[link][metric] for metric in metrics] for link in data])
    
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
    data: Dict[Tuple[str, str], Dict[str, float]], 
    weights: Dict[str, float]
) -> Dict[Tuple[str, str], float]:
    """
    使用TOPSIS法计算链路综合得分
    :param data: 标准化后的链路指标
    :param weights: 指标权重
    :return: 链路得分字典
    """
    if not data or not weights:
        raise ValueError("输入数据或权重不能为空")
    
    metrics = list(next(iter(data.values())).keys())
    normalized_array = np.array([[data[link][metric] for metric in metrics] for link in data])
    
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

def identify_key_links(
    file_path: str
) -> List[Tuple[Tuple[str, str], float]]:
    """
    从文件中读取链路指标并识别关键链路（前20%）
    :param file_path: 链路指标文件路径
    :return: 关键链路列表，包含链路对和得分
    """
    # 读取数据
    link_metrics = read_link_metrics(file_path)
    if not link_metrics:
        return []
    
    # 数据标准化
    normalized_metrics = normalize_data(link_metrics)
    # 计算权重
    weights = entropy_weight_method(normalized_metrics)
    # 计算得分
    scores = topsis_method(normalized_metrics, weights)
    
    # 选取前20%的链路
    k = math.ceil(len(link_metrics) * 0.2)
    key_links = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return key_links

def calculate_link_scores(
    file_path: str
) -> Dict[Tuple[str, str], float]:
    """
    计算所有链路的得分
    :param file_path: 链路指标文件路径
    :return: 链路得分字典
    """
    link_metrics = read_link_metrics(file_path)
    normalized_metrics = normalize_data(link_metrics)
    weights = entropy_weight_method(normalized_metrics)
    return topsis_method(normalized_metrics, weights)

def identify_key_links_from_dict(
    metrics_dict: Dict[Tuple[str, str], Dict[str, float]]
) -> List[Tuple[Tuple[str, str], float]]:
    """
    从字典直接识别关键链路（不依赖文件）
    :param metrics_dict: 链路指标字典
    :return: 关键链路列表
    """
    normalized_metrics = normalize_data(metrics_dict)
    weights = entropy_weight_method(normalized_metrics)
    scores = topsis_method(normalized_metrics, weights)
    
    k = math.ceil(len(metrics_dict) * 0.2)
    key_links = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return key_links

def save_key_links_to_file(key_links: List[Tuple[Tuple[str, str], float]], output_file: str):
    """将关键链路保存到文件"""
    with open(output_file, 'w') as f:
        # f.write(f"# 关键链路（前20%，共{len(key_links)}条）\n")
        for i, (link, score) in enumerate(key_links, 1):
            f.write(f"链路 {link}: 得分 = {score:.6f}\n")
    print(f"关键链路已保存至 {output_file}")

def main():
    input_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/link_metrics.txt"  # 输入文件路径
    output_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/critical_links_only.txt"  # 输出文件路径
    
    try:
        key_links = identify_key_links(input_file)
        if key_links:
            save_key_links_to_file(key_links, output_file)
            print(f"成功识别{len(key_links)}条关键链路")
        else:
            print("未识别到关键链路")
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    main()