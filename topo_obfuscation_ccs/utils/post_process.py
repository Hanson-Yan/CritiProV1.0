# post_process.py - 后处理筛选模块

import numpy as np
from typing import List, Tuple, Dict
from utils.critical_node import identify_key_nodes_from_dict
from utils.metrics import VirtualNodeMetrics


def calculate_overlap(matrix: np.ndarray, 
                      key_nodes: List[str], 
                      node_metrics: Dict,
                      encoder) -> int:
    """
    计算关键节点重合度
    
    参数:
        matrix: 邻接矩阵
        key_nodes: 原始关键节点列表
        node_metrics: 节点度量字典
        encoder: 编码器（未使用，保留接口）
    
    返回:
        overlap: 重合节点数量（失败时返回999）
    """
    try:
        metrics_gen = VirtualNodeMetrics(matrix, node_metrics)
        virtual_metrics = metrics_gen.get_metrics()
        key_nodes_new = identify_key_nodes_from_dict(virtual_metrics)
        
        if not isinstance(key_nodes_new, list):
            return 999
        
        current_keys = [str(node) for node, _ in key_nodes_new]
        overlap = len(set(key_nodes) & set(current_keys))
        
        return overlap
        
    except Exception as e:
        print(f"[警告] 重合度计算失败: {e}")
        return 999


def filter_solutions_by_overlap(
    pareto_front: List,
    encoder,
    key_nodes: List[str],
    node_metrics: Dict,
    max_overlap: int = 1
) -> Tuple[List, int, Dict]:
    """
    从 Pareto 前沿中筛选满足重合度要求的解
    
    参数:
        pareto_front: Pareto 最优解列表
        encoder: 编码器
        key_nodes: 原始关键节点
        node_metrics: 节点度量
        max_overlap: 最大允许重合度（默认1）
    
    返回:
        selected_solutions: 筛选后的解
        actual_overlap: 实际达到的重合度
        statistics: 统计信息字典
    """
    print(f"\n{'='*70}")
    print("后处理筛选阶段")
    print(f"{'='*70}")
    print(f"待筛选解数量: {len(pareto_front)}")
    print(f"最大允许重合: {max_overlap}")
    
    solutions_by_overlap = {}
    
    # 对每个解计算重合度
    for idx, solution in enumerate(pareto_front):
        if (idx + 1) % 20 == 0:
            print(f"  筛选进度: {idx + 1}/{len(pareto_front)}")
        
        matrix = encoder.decode(solution)
        overlap = calculate_overlap(matrix, key_nodes, node_metrics, encoder)
        
        # 分组存储
        if overlap not in solutions_by_overlap:
            solutions_by_overlap[overlap] = []
        solutions_by_overlap[overlap].append(solution)
    
    # 统计信息
    statistics = {
        'total': len(pareto_front),
        'by_overlap': {k: len(v) for k, v in solutions_by_overlap.items()},
        'valid_overlaps': sorted([k for k in solutions_by_overlap.keys() if k < 999])
    }
    
    # 打印统计
    print(f"\n重合度分布:")
    for overlap in sorted(solutions_by_overlap.keys()):
        count = len(solutions_by_overlap[overlap])
        percentage = count / len(pareto_front) * 100
        marker = "✓" if overlap <= max_overlap else "✗"
        print(f"  [{marker}] 重合 {overlap:2d} 个: {count:4d} ({percentage:5.1f}%)")
    
    # 按优先级返回
    for level in range(max_overlap + 1):
        if level in solutions_by_overlap and len(solutions_by_overlap[level]) > 0:
            selected = solutions_by_overlap[level]
            print(f"\n{'='*70}")
            print(f"[✓] 筛选成功")
            print(f"    重合度: {level}")
            print(f"    解数量: {len(selected)}")
            print(f"{'='*70}")
            return selected, level, statistics
    
    # 如果没有满足条件的，返回最接近的
    available_levels = [k for k in solutions_by_overlap.keys() if k < 999]
    if available_levels:
        best_level = min(available_levels)
        selected = solutions_by_overlap[best_level]
        print(f"\n{'='*70}")
        print(f"[⚠] 未找到重合度 ≤ {max_overlap} 的解")
        print(f"    返回最优重合度: {best_level}")
        print(f"    解数量: {len(selected)}")
        print(f"{'='*70}")
        return selected, best_level, statistics
    
    # 完全无解
    print(f"\n{'='*70}")
    print(f"[✗] 错误：未找到任何有效解")
    print(f"{'='*70}")
    return [], -1, statistics


def select_best_from_filtered(
    filtered_solutions: List,
    evaluator
) -> Tuple[List[int], Tuple[float, float, float]]:
    """
    从筛选后的解中选择最优解
    
    优先级: 
        1. 最小化 key_score
        2. 最大化 similarity
        3. 最小化 penalty
    
    参数:
        filtered_solutions: 筛选后的解列表
        evaluator: 评估器
    
    返回:
        best_individual: 最优个体
        best_fitness: 最优适应度 (key_score, similarity, penalty)
    """
    if len(filtered_solutions) == 0:
        raise ValueError("筛选后的解集为空")
    
    print(f"\n从 {len(filtered_solutions)} 个解中选择最优解...")
    
    # 重新评估所有解（确保准确性）
    evaluated = []
    for solution in filtered_solutions:
        fitness = evaluator.evaluate(solution)
        evaluated.append((solution, fitness))
    
    # 过滤无效解
    valid = [x for x in evaluated if x[1] not in [(1.0, 0.0, 1000.0), (1.0, 0.0, 100.0)]]
    
    if len(valid) == 0:
        print("[警告] 所有解都无效，使用原始解")
        valid = evaluated
    
    # 排序：目标1升序，目标2降序，目标3升序
    best = sorted(valid, key=lambda x: (x[1][0], -x[1][1], x[1][2]))[0]
    
    print(f"最优解适应度:")
    print(f"  关键节点得分: {best[1][0]:.4f}")
    print(f"  成本: {best[1][1]:.4f}")
    print(f"  惩罚值: {best[1][2]:.4f}")
    
    return best[0], best[1]


def print_final_analysis(
    best_individual: List[int],
    encoder,
    key_nodes: List[str],
    node_metrics: Dict,
    overlap_level: int
):
    """
    打印最终分析结果
    
    参数:
        best_individual: 最优个体
        encoder: 编码器
        key_nodes: 原始关键节点
        node_metrics: 节点度量
        overlap_level: 重合度
    """
    matrix = encoder.decode(best_individual)
    
    try:
        metrics_gen = VirtualNodeMetrics(matrix, node_metrics)
        virtual_metrics = metrics_gen.get_metrics()
        key_nodes_final = [str(node) for node, _ in identify_key_nodes_from_dict(virtual_metrics)]
        overlap_nodes = set(key_nodes) & set(key_nodes_final)
    except:
        key_nodes_final = []
        overlap_nodes = set()
    
    print(f"\n{'='*70}")
    print("最终解分析")
    print(f"{'='*70}")
    print(f"原始关键节点: {key_nodes}")
    print(f"混淆关键节点: {key_nodes_final}")
    print(f"重合节点: {list(overlap_nodes)} (数量: {len(overlap_nodes)})")
    
    if overlap_level == 0:
        print(f"\n[✓✓✓] 完美：实现了完全无重合的关键节点转移")
    elif overlap_level == 1:
        print(f"\n[✓✓] 良好：仅1个关键节点重合（隐藏率: {(1 - 1/len(key_nodes))*100:.1f}%）")
    elif overlap_level == 2:
        print(f"\n[✓] 可接受：2个关键节点重合（隐藏率: {(1 - 2/len(key_nodes))*100:.1f}%）")
    else:
        print(f"\n[⚠] 次优：{overlap_level}个关键节点重合（隐藏率: {(1 - overlap_level/len(key_nodes))*100:.1f}%）")
    
    print(f"{'='*70}\n")
