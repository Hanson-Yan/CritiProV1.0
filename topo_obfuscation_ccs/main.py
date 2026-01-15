# main.py - 完整版（相似度软约束 + 保存Pareto前沿解）

import numpy as np
from encoder import SparseEdgeEncoder
from objective import ObfuscationObjective
from nsga2_solver import NSGA2Solver
from hill_climb import local_hill_climb
from parallel_utils import parallel_evaluate

from utils.critical_node import identify_key_nodes, identify_key_nodes_from_dict
from utils.metrics import VirtualNodeMetrics
from utils.similarity import graph_similarity
from draw_topo import DrawTopology

from utils.post_process import (
    filter_solutions_by_overlap,
    select_best_from_filtered,
    print_final_analysis
)

import os
import time
import pickle
from functools import partial


def copy_file(source_file, target_file):
    try:
        with open(source_file, 'r') as src:
            content = src.read()
        with open(target_file, 'w') as tgt:
            tgt.write(content)
        print(f"成功将 {source_file} 的内容拷贝到 {target_file}")
    except FileNotFoundError:
        print(f"错误：文件 {source_file} 未找到")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"发生错误：{e}")


def prepare_data(topo_num):
    Criti_pro_dir = "/home/retr0/Project/TopologyObfu/CritiPro/"
    topo_matrix = "/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/"
    topo_num_result = f"{topo_num}_result/"
    topo_num_txt = f"{topo_num}.txt"
    metrics_txt = "metrics.txt"

    topo_obfuscation_data = Criti_pro_dir + "topo_obfuscation_ccs/" + "data/"
    input_adj_txt = topo_obfuscation_data + "input_adj.txt"
    input_metrics_txt = topo_obfuscation_data + "input_metrics.txt"

    copy_file(topo_matrix + topo_num_txt, input_adj_txt)
    copy_file(Criti_pro_dir + topo_num_result + metrics_txt, input_metrics_txt)

def load_adjacency_matrix(file_path: str):
    return np.loadtxt(file_path)


def load_node_metrics(file_path: str):
    return file_path


def auto_select_bhop(n_nodes: int) -> int:
    """自动选择 b-hop"""
    if n_nodes <= 50:
        return 2
    elif n_nodes <= 150:
        return 3
    else:
        return 4

def auto_config_nsga2(n_nodes: int, n_edges: int):
    """根据拓扑规模自适应配置 NSGA-II 参数"""
    complexity_score = n_nodes + n_edges / 2
    
    if n_nodes <= 20 or complexity_score <= 35:
        return {
            'population_size': 100,
            'generations': 100,
            'max_workers': 6,
            'crossover_prob': 0.9,
            'mutation_prob': 0.2
        }
    elif n_nodes <= 40 or complexity_score <= 80:
        return {
            'population_size': 150,
            'generations': 150,
            'max_workers': 8,
            'crossover_prob': 0.9,
            'mutation_prob': 0.2
        }
    elif n_nodes <= 70 or complexity_score <= 150:
        return {
            'population_size': 200,
            'generations': 200,
            'max_workers': 10,
            'crossover_prob': 0.85,
            'mutation_prob': 0.25
        }
    else:
        return {
            'population_size': 250,
            'generations': 250,
            'max_workers': 12,
            'crossover_prob': 0.85,
            'mutation_prob': 0.3
        }


def print_config_info(config: dict, n_nodes: int, n_edges: int, b_hop: int):
    """打印配置信息"""
    print(f"\n{'='*70}")
    print(f"拓扑信息与自适应配置")
    print(f"{'='*70}")
    print(f"[拓扑规模]")
    print(f"  节点数: {n_nodes}")
    print(f"  边数: {n_edges}")
    print(f"  复杂度评分: {n_nodes + n_edges/2:.1f}")
    print(f"  自动选定 b-hop: {b_hop}")
    
    print(f"\n[NSGA-II 配置]")
    print(f"  种群大小: {config['population_size']}")
    print(f"  迭代代数: {config['generations']}")
    print(f"  交叉概率: {config['crossover_prob']}")
    print(f"  变异概率: {config['mutation_prob']}")
    
    print(f"\n[优化策略]")
    print(f"  执行模式: 完整迭代（无早停）")
    print(f"  约束模式: 相似度软约束（惩罚函数法）")
    print(f"  说明: 完整执行 {config['generations']} 代进化，")
    print(f"        允许探索相似度区间外的解（通过惩罚引导）")
    
    print(f"\n[并行配置]")
    print(f"  并行进程数: {config['max_workers']}")
    
    print(f"{'='*70}\n")


def save_pareto_front(pareto_front, encoder, key_nodes, node_metrics, 
                      evaluator, save_path, b_hop):
    """保存 Pareto 前沿"""
    n_nodes = encoder.original_matrix.shape[0]
    n_edges = int(np.sum(encoder.original_matrix) / 2)
    
    pareto_data = {
        'meta': {
            'n_solutions': len(pareto_front),
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'b_hop': b_hop,
            'key_nodes': key_nodes,
            'similarity_range': [0.6, 0.9],
            'constraint_mode': 'soft',  # ← 标记为软约束
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'solutions': []
    }
    
    print(f"\n{'='*70}")
    print(f"保存 Pareto 前沿解集")
    print(f"{'='*70}")
    print(f"解集数量: {len(pareto_front)}")
    
    for idx, individual in enumerate(pareto_front):
        matrix = encoder.decode(individual)
        
        # 识别关键节点
        try:
            metrics_gen = VirtualNodeMetrics(matrix, node_metrics)
            virtual_metrics = metrics_gen.get_metrics()
            key_nodes_conf = [node for node, _ in identify_key_nodes_from_dict(virtual_metrics)]
        except:
            key_nodes_conf = []
        
        overlap = len(set(key_nodes) & set(key_nodes_conf))
        
        # 计算成本
        original_vec = encoder.encode(encoder.original_matrix)
        conf_vec = encoder.encode(matrix)
        cost = sum(o != c for o, c in zip(original_vec, conf_vec))
        
        # 计算相似度
        similarity = graph_similarity(encoder.original_matrix, matrix, method='portrait')
        
        # 适应度
        key_score, normalized_cost, penalty = individual.fitness.values
        
        solution_data = {
            'index': idx,
            'individual': individual,
            'matrix': matrix,
            'fitness': {
                'key_score': key_score,
                'normalized_cost': normalized_cost,
                'actual_cost': cost,
                'penalty': penalty
            },
            'metrics': {
                'similarity': similarity,
                'similarity_satisfied': 0.6 <= similarity <= 0.9,  # ← 标记是否满足
                'key_nodes_original': key_nodes,
                'key_nodes_confused': key_nodes_conf,
                'overlap': overlap,
                'overlap_ratio': overlap / len(key_nodes) if len(key_nodes) > 0 else 0
            }
        }
        
        pareto_data['solutions'].append(solution_data)
        
        if (idx + 1) % 10 == 0:
            print(f"  处理: {idx + 1}/{len(pareto_front)}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(pareto_data, f)
    
    print(f"✓ 已保存到: {save_path}")
    print(f"{'='*70}\n")
    
    return pareto_data

if __name__ == "__main__":
    topo_num = input("请输入拓扑编号: ")
    prepare_data(topo_num)
    
    # 输入路径
    adj_path = "data/input_adj.txt"
    metrics_path = "data/input_metrics.txt"

    topo_num_output_file = f"data/{topo_num}_output_file/"
    if not os.path.exists(topo_num_output_file):
        os.mkdir(topo_num_output_file)
    
    result_txt_path = f"{topo_num_output_file}output_adj.txt"
    result_png_path = f"{topo_num_output_file}confuse_topo.png"
    result_key_node_path = f"{topo_num_output_file}critinode_critipro.txt"
    pareto_save_path = f"{topo_num_output_file}pareto_front.pkl"

    confuse_st = time.perf_counter()
    
    # ========== 加载数据 ==========
    matrix = load_adjacency_matrix(adj_path)
    metrics = load_node_metrics(metrics_path)

    n_nodes = matrix.shape[0]
    n_edges = int(np.sum(matrix) / 2)
    b_hop = auto_select_bhop(n_nodes)
    
    # ========== 自适应配置 ==========
    config = auto_config_nsga2(n_nodes, n_edges)
    print_config_info(config, n_nodes, n_edges, b_hop)
    
    # ========== 识别关键节点 ==========
    key_nodes_full = identify_key_nodes(metrics)
    key_nodes = [node for node, _ in key_nodes_full]
    
    print(f"关键节点（原始拓扑）: {key_nodes}")
    print(f"关键节点数量: {len(key_nodes)}")

    # ========== 编码器 ==========
    encoder = SparseEdgeEncoder(n_nodes=n_nodes, original_matrix=matrix, b_hop=b_hop)

    # ========== 创建评估器（相似度软约束）==========
    evaluator = ObfuscationObjective(
        original_matrix=matrix,
        key_nodes=key_nodes,
        node_metrics=metrics,
        encoder=encoder,
        b_hop=b_hop,
        alpha_min=0.6,
        alpha_max=0.9,
        similarity_penalty_weight=10.0,  # ← 新增：相似度惩罚权重
        latency_penalty_weight=10.0,
        overlap_mode='soft',
        overlap_penalty_weight=100.0,
        adaptive_overlap=False,
        debug_mode=False
    )
    
    # ========== NSGA-II 求解 ==========
    solver = NSGA2Solver(
        evaluator=evaluator,
        individual_length=len(encoder.get_edge_list()),
        population_size=config['population_size'],
        generations=config['generations'],
        # generations=100,#test time
        crossover_prob=config['crossover_prob'],
        mutation_prob=config['mutation_prob']
    )
    
    parallel_eval_fn = partial(
        parallel_evaluate, 
        evaluator=evaluator, 
        max_workers=config['max_workers']
    )
    
    print("\n开始 NSGA-II 优化...")
    print(f"提示: 进化过程中每 10 代会输出前3个最优有效解\n")
    
    hof = solver.run(parallel_evaluate_fn=parallel_eval_fn)
    
    if len(hof) == 0:
        print(f"\n{'='*70}")
        print("[✗] 未找到任何解")
        print(f"{'='*70}")
        exit()
    
    print(f"\nNSGA-II 输出 {len(hof)} 个 Pareto 最优解")
    
    # ========== 保存 Pareto 前沿解集 ==========
    pareto_data = save_pareto_front(
        pareto_front=hof,
        encoder=encoder,
        key_nodes=key_nodes,
        node_metrics=metrics,
        evaluator=evaluator,
        save_path=pareto_save_path,
        b_hop=b_hop
    )
    
    # ========== 后处理筛选 ==========
    print(f"\n{'='*70}")
    print("后处理筛选")
    print(f"{'='*70}")
    
    filtered_solutions, overlap_level, filter_stats = filter_solutions_by_overlap(
        pareto_front=hof,
        encoder=encoder,
        key_nodes=key_nodes,
        node_metrics=metrics,
        max_overlap=1
    )
    
    print(f"筛选前: {len(hof)} 个解")
    print(f"筛选后: {len(filtered_solutions)} 个解")
    print(f"重合度统计: {filter_stats}")
    
    if len(filtered_solutions) == 0:
        print("[✗] 后处理筛选失败：无有效解")
        print("建议: 调整 max_overlap 参数或检查拓扑特性")
        exit()
    
    # ========== 局部爬山微调 ==========
    print(f"\n{'='*70}")
    print("局部优化")
    print(f"{'='*70}")
    print(f"对 {len(filtered_solutions)} 个筛选解进行微调...")
    
    refined = []
    for i, ind in enumerate(filtered_solutions):
        if i % 10 == 0 and i > 0:
            print(f"  微调进度: {i}/{len(filtered_solutions)}")
        refined_ind = local_hill_climb(ind, evaluator, max_steps=5)
        refined.append(refined_ind)
    
    print(f"微调完成")
    
    # ========== 选择最优解 ==========
    print(f"\n{'='*70}")
    print("最优解选择")
    print(f"{'='*70}")
    
    best_individual, best_fitness = select_best_from_filtered(
        filtered_solutions=refined,
        evaluator=evaluator
    )
    
    # ========== 分析最终结果 ==========
    print_final_analysis(
        best_individual=best_individual,
        encoder=encoder,
        key_nodes=key_nodes,
        node_metrics=metrics,
        overlap_level=overlap_level
    )
    
    # ========== 提取最终信息 ==========
    final_matrix = encoder.decode(best_individual)
    key_score, normalized_cost, penalty = best_fitness
    
    try:
        metrics_gen = VirtualNodeMetrics(final_matrix, metrics)
        virtual_metrics = metrics_gen.get_metrics()
        key_nodes_final = [node for node, _ in identify_key_nodes_from_dict(virtual_metrics)]
    except:
        key_nodes_final = []
    
    overlap_nodes = set(key_nodes) & set(key_nodes_final)
    
    # 计算实际修改成本
    original_vec = encoder.encode(matrix)
    final_vec = encoder.encode(final_matrix)
    actual_cost = sum(o != f for o, f in zip(original_vec, final_vec))
    
    # ========== 打印最终总结 ==========
    print(f"\n{'='*70}")
    print(f"最终结果总结")
    print(f"{'='*70}")
    
    print(f"\n[适应度]")
    print(f"  关键节点平均得分: {key_score:.4f}")
    print(f"  归一化成本: {normalized_cost:.4f}")
    print(f"  实际修改边数: {actual_cost}")
    print(f"  约束惩罚: {penalty:.4f}")
    
    print(f"\n[拓扑相似度]")
    final_similarity = graph_similarity(matrix, final_matrix, method='portrait')
    print(f"  相似度: {final_similarity:.4f}")
    print(f"  约束范围: [0.6, 0.9]")
    print(f"  是否满足: {'是' if 0.6 <= final_similarity <= 0.9 else '否'}")
    if not (0.6 <= final_similarity <= 0.9):
        deviation = min(abs(final_similarity - 0.6), abs(final_similarity - 0.9))
        print(f"  偏离量: {deviation:.4f}")
    
    # ========== 保存结果 ==========
    print("\n保存结果...")
    np.savetxt(result_txt_path, final_matrix, fmt="%d")
    
    if len(key_nodes_final) > 0:
        key_node_numbers = [int(''.join(filter(str.isdigit, item))) for item in key_nodes_final]
        np.savetxt(result_key_node_path, key_node_numbers, fmt="%d")
    
    confuse_et = time.perf_counter()
    elapsed_time = confuse_et - confuse_st
    
    print(f"\n{'='*70}")
    print(f"性能统计")
    print(f"{'='*70}")
    print(f"混淆求解总时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    final_stats = evaluator.get_statistics()
    print(f"\n评估统计:")
    print(f"  总评估次数: {final_stats['evaluations']}")
    print(f"  可行解数量: {final_stats['feasible_count']}")
    print(f"  可行解比例: {final_stats['feasible_ratio']:.2%}")
    print(f"  平均每次评估耗时: {elapsed_time/final_stats['evaluations']*1000:.2f} 毫秒")
    
    print(f"{'='*70}")
    
    # ========== 可视化 ==========
    print("\n生成可视化...")
    DrawTopology(matrix=matrix, critical_nodes=key_nodes).draw(show=True)
    DrawTopology(matrix=final_matrix, critical_nodes=key_nodes_final).draw(
        show=True, save_path=result_png_path
    )
    
    print(f"\n结果已保存到: {topo_num_output_file}")
    print(f"  - 邻接矩阵: {result_txt_path}")
    print(f"  - 关键节点: {result_key_node_path}")
    print(f"  - 可视化图: {result_png_path}")
    print(f"  - Pareto前沿: {pareto_save_path}")
    
    print(f"\n{'='*70}")
    print("混淆完成！")
    print(f"{'='*70}\n")
