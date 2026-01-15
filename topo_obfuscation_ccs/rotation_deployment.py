# rotation_deployment.py - 混淆拓扑轮换动态部署

import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt  # ← 添加这一行
from draw_topo import DrawTopology

def draw_topology_non_blocking(matrix, critical_nodes, save_path=None, show=True):
    """
    非阻塞方式绘制网络拓扑图（专用于轮换部署）
    
    Args:
        matrix: 邻接矩阵
        critical_nodes: 关键节点列表（可以是索引 [0,1,2] 或字符串 ["s0","s1","s2"]）
        save_path: 保存路径
        show: 是否显示图像
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 关闭之前的图形
    plt.close('all')
    
    # 创建新图形
    plt.figure(figsize=(10, 8))
    
    # 构建图
    G = nx.Graph()
    num_nodes = len(matrix)
    nodes = [f"s{i}" for i in range(num_nodes)]
    
    # 添加节点
    for node in nodes:
        G.add_node(node, color='lightgreen', shape='o')
    
    # 添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if matrix[i][j] > 0:
                G.add_edge(f"s{i}", f"s{j}")
    
    # ======== 修正：智能判断关键节点格式 ========
    if critical_nodes:
        # 判断是否已经是 sX 格式
        if isinstance(critical_nodes[0], str):
            critical_nodes_formatted = critical_nodes  # 已经是 "s0", "s1" 格式
        else:
            critical_nodes_formatted = [f"s{i}" for i in critical_nodes]  # 是索引格式
    else:
        critical_nodes_formatted = []
    # =========================================
    
    # 绘制
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制普通节点
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_size = 200
    font_size = 8
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), 
                          node_color=node_colors, node_shape='o', node_size=node_size)
    
    # 突出显示关键节点
    if critical_nodes_formatted:
        nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes_formatted, 
                              node_color='red', node_shape='o', 
                              node_size=node_size+50, edgecolors='black', linewidths=2)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos)
    
    # 添加标签
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    
    # 设置图形样式
    plt.style.use('default')
    plt.box(False)
    plt.title("Network Topology - Rotation Deployment")
    plt.axis('off')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, format='png', dpi=600)
    
    # 非阻塞显示
    if show:
        plt.draw()
        plt.pause(0.1)
    else:
        plt.close()



def load_pareto_front(pareto_file):
    """
    加载 Pareto 前沿解集
    
    Args:
        pareto_file: Pareto 前沿数据文件路径
    
    Returns:
        pareto_data: Pareto 前沿数据字典
    """
    print(f"\n{'='*70}")
    print(f"加载 Pareto 前沿解集")
    print(f"{'='*70}")
    print(f"文件路径: {pareto_file}")
    
    if not os.path.exists(pareto_file):
        raise FileNotFoundError(f"Pareto 前沿文件不存在: {pareto_file}")
    
    with open(pareto_file, 'rb') as f:
        pareto_data = pickle.load(f)
    
    meta = pareto_data['meta']
    solutions = pareto_data['solutions']
    
    print(f"\n[元信息]")
    print(f"  拓扑节点数: {meta['n_nodes']}")
    print(f"  拓扑边数: {meta['n_edges']}")
    print(f"  b-hop 限制: {meta['b_hop']}")
    print(f"  原始关键节点: {meta['key_nodes']}")
    print(f"  成本阈值: {meta['cost_threshold']}")
    print(f"  生成时间: {meta['timestamp']}")
    
    print(f"\n[解集信息]")
    print(f"  Pareto 前沿解数量: {meta['n_solutions']}")
    print(f"  实际加载解数量: {len(solutions)}")
    
    print(f"{'='*70}\n")
    
    return pareto_data


def filter_solutions_by_criteria(pareto_data, max_overlap=None, 
                                  max_cost_violation=None, 
                                  min_similarity=None):
    """
    根据条件筛选解
    
    Args:
        pareto_data: Pareto 前沿数据
        max_overlap: 最大重合度（None 表示不限制）
        max_cost_violation: 最大成本超限量（None 表示不限制）
        min_similarity: 最小相似度（None 表示不限制）
    
    Returns:
        filtered_solutions: 筛选后的解列表
    """
    solutions = pareto_data['solutions']
    filtered = []
    
    print(f"\n{'='*70}")
    print(f"筛选 Pareto 前沿解")
    print(f"{'='*70}")
    
    criteria = []
    if max_overlap is not None:
        criteria.append(f"重合度 ≤ {max_overlap}")
    if max_cost_violation is not None:
        criteria.append(f"成本超限 ≤ {max_cost_violation}")
    if min_similarity is not None:
        criteria.append(f"相似度 ≥ {min_similarity}")
    
    print(f"筛选条件: {' AND '.join(criteria) if criteria else '无（使用全部解）'}")
    
    for sol in solutions:
        metrics = sol['metrics']
        fitness = sol['fitness']
        
        # 检查各项条件
        pass_overlap = (max_overlap is None or 
                        metrics['overlap'] <= max_overlap)
        pass_cost = (max_cost_violation is None or 
                     metrics['cost_violation'] <= max_cost_violation)
        pass_sim = (min_similarity is None or 
                    fitness['similarity'] >= min_similarity)
        
        if pass_overlap and pass_cost and pass_sim:
            filtered.append(sol)
    
    print(f"筛选前: {len(solutions)} 个解")
    print(f"筛选后: {len(filtered)} 个解")
    print(f"{'='*70}\n")
    
    return filtered


def print_solution_summary(solution, index, total):
    """
    打印单个解的摘要信息
    
    Args:
        solution: 解数据
        index: 当前索引（从1开始）
        total: 总解数
    """
    metrics = solution['metrics']
    fitness = solution['fitness']
    
    print(f"\n{'='*70}")
    print(f"方案 {index}/{total} - 详细信息")
    print(f"{'='*70}")
    
    print(f"\n[适应度]")
    print(f"  关键节点平均得分: {fitness['key_score']:.4f}")
    print(f"  拓扑相似度: {fitness['similarity']:.4f}")
    print(f"  约束惩罚: {fitness['penalty']:.4f}")
    
    print(f"\n[关键节点]")
    print(f"  原始关键节点: {metrics['key_nodes_original']}")
    print(f"  混淆关键节点: {metrics['key_nodes_confused']}")
    overlap_nodes = set(metrics['key_nodes_original']) & set(metrics['key_nodes_confused'])
    print(f"  重合节点: {list(overlap_nodes)}")
    print(f"  重合度: {metrics['overlap']}/{len(metrics['key_nodes_original'])} "
          f"({metrics['overlap_ratio']*100:.1f}%)")
    
    print(f"\n[扰动成本]")
    print(f"  实际修改边数: {metrics['cost']}")
    print(f"  成本阈值: {metrics['cost_threshold']}")
    print(f"  是否超限: {'是' if metrics['cost_violation'] > 0 else '否'}")
    if metrics['cost_violation'] > 0:
        print(f"  超限量: {metrics['cost_violation']}")
    
    print(f"{'='*70}\n")


def rotation_deploy(pareto_file, output_dir, 
                    max_overlap=None, 
                    max_cost_violation=None,
                    min_similarity=None,
                    rotation_interval=5.0,
                    max_rotations=None,
                    save_images=True,
                    show_plots=True):
    """
    混淆拓扑轮换动态部署
    
    Args:
        pareto_file: Pareto 前沿数据文件路径
        output_dir: 输出目录
        max_overlap: 最大重合度筛选条件
        max_cost_violation: 最大成本超限量筛选条件
        min_similarity: 最小相似度筛选条件
        rotation_interval: 轮换间隔（秒）
        max_rotations: 最大轮换次数（None 表示无限循环）
        save_images: 是否保存图像
        show_plots: 是否显示图像
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载 Pareto 前沿解集
    pareto_data = load_pareto_front(pareto_file)
    
    # 筛选解
    solutions = filter_solutions_by_criteria(
        pareto_data=pareto_data,
        max_overlap=max_overlap,
        max_cost_violation=max_cost_violation,
        min_similarity=min_similarity
    )
    
    if len(solutions) == 0:
        print("[✗] 筛选后无可用解，请放宽筛选条件")
        return
    
    # 提取元信息
    original_key_nodes = pareto_data['meta']['key_nodes']
    
    print(f"\n{'='*70}")
    print(f"开始轮换部署")
    print(f"{'='*70}")
    print(f"可用方案数: {len(solutions)}")
    print(f"轮换间隔: {rotation_interval} 秒")
    print(f"最大轮换次数: {max_rotations if max_rotations else '无限'}")
    print(f"输出目录: {output_dir}")
    print(f"保存图像: {'是' if save_images else '否'}")
    print(f"显示图像: {'是' if show_plots else '否'}")
    print(f"{'='*70}\n")
    
    rotation_count = 0
    
    try:
        while True:
            for idx, solution in enumerate(solutions, start=1):
                # 检查是否达到最大轮换次数
                if max_rotations is not None and rotation_count >= max_rotations:
                    print(f"\n已达到最大轮换次数 {max_rotations}，停止部署")
                    return
                
                rotation_count += 1
                
                # 提取数据
                matrix = solution['matrix']
                key_nodes_conf = solution['metrics']['key_nodes_confused']
                
                # 打印当前方案信息
                print(f"\n{'='*70}")
                print(f"[轮换 {rotation_count}] 部署方案 {idx}/{len(solutions)}")
                print(f"{'='*70}")
                print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 打印详细信息
                print_solution_summary(solution, idx, len(solutions))
                
                # 生成文件名
                if save_images:
                    save_path = os.path.join(
                        output_dir, 
                        f"rotation_{rotation_count:04d}_solution_{idx:03d}.png"
                    )
                else:
                    save_path = None
                
                # ======== 关键修改：关闭之前的窗口 ========
                # 绘制拓扑（使用非阻塞函数）
                print(f"正在绘制拓扑可视化...")
                draw_topology_non_blocking(
                    matrix=matrix,
                    critical_nodes=key_nodes_conf,
                    save_path=save_path,
                    show=show_plots
                )
                
                if save_path:
                    print(f"✓ 已保存: {save_path}")
                
                # 保存邻接矩阵
                if save_images:
                    matrix_path = os.path.join(
                        output_dir,
                        f"rotation_{rotation_count:04d}_solution_{idx:03d}_adj.txt"
                    )
                    np.savetxt(matrix_path, matrix, fmt="%d")
                    print(f"✓ 已保存邻接矩阵: {matrix_path}")
                
                # ======== 关键修改：等待轮换间隔 ========
                if idx < len(solutions) or max_rotations is None:
                    print(f"\n等待 {rotation_interval} 秒后切换到下一方案...")
                    if show_plots:
                        plt.pause(rotation_interval)  # 显示图像时使用 plt.pause
                    else:
                        time.sleep(rotation_interval)  # 不显示时使用 time.sleep
            
            # 一轮完成后
            if max_rotations is None:
                print(f"\n{'='*70}")
                print(f"完成一轮轮换（共 {len(solutions)} 个方案）")
                print(f"开始新一轮...")
                print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"用户中断轮换部署")
        print(f"{'='*70}")
        print(f"已完成轮换次数: {rotation_count}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*70}\n")
    
    finally:
        # ======== 关键修改：结束时关闭所有窗口 ========
        if show_plots:
            plt.close('all')


def batch_export_all_solutions(pareto_file, output_dir, 
                                max_overlap=None,
                                max_cost_violation=None,
                                min_similarity=None):
    """
    批量导出所有符合条件的解（不轮换显示）
    
    Args:
        pareto_file: Pareto 前沿数据文件路径
        output_dir: 输出目录
        max_overlap: 最大重合度筛选条件
        max_cost_violation: 最大成本超限量筛选条件
        min_similarity: 最小相似度筛选条件
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载 Pareto 前沿解集
    pareto_data = load_pareto_front(pareto_file)
    
    # 筛选解
    solutions = filter_solutions_by_criteria(
        pareto_data=pareto_data,
        max_overlap=max_overlap,
        max_cost_violation=max_cost_violation,
        min_similarity=min_similarity
    )
    
    if len(solutions) == 0:
        print("[✗] 筛选后无可用解，请放宽筛选条件")
        return
    
    print(f"\n{'='*70}")
    print(f"批量导出所有解")
    print(f"{'='*70}")
    print(f"解数量: {len(solutions)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    # 保存摘要信息
    summary_path = os.path.join(output_dir, "solutions_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"Pareto 前沿解集摘要\n")
        f.write(f"{'='*70}\n")
        f.write(f"生成时间: {pareto_data['meta']['timestamp']}\n")
        f.write(f"拓扑节点数: {pareto_data['meta']['n_nodes']}\n")
        f.write(f"拓扑边数: {pareto_data['meta']['n_edges']}\n")
        f.write(f"原始关键节点: {pareto_data['meta']['key_nodes']}\n")
        f.write(f"成本阈值: {pareto_data['meta']['cost_threshold']}\n")
        f.write(f"解数量: {len(solutions)}\n")
        f.write(f"{'='*70}\n\n")
        
        for idx, solution in enumerate(solutions, start=1):
            metrics = solution['metrics']
            fitness = solution['fitness']
            
            f.write(f"\n{'='*70}\n")
            f.write(f"方案 {idx}/{len(solutions)}\n")
            f.write(f"{'='*70}\n")
            
            f.write(f"\n[适应度]\n")
            f.write(f"  关键节点平均得分: {fitness['key_score']:.4f}\n")
            f.write(f"  拓扑相似度: {fitness['similarity']:.4f}\n")
            f.write(f"  约束惩罚: {fitness['penalty']:.4f}\n")
            
            f.write(f"\n[关键节点]\n")
            f.write(f"  原始关键节点: {metrics['key_nodes_original']}\n")
            f.write(f"  混淆关键节点: {metrics['key_nodes_confused']}\n")
            overlap_nodes = set(metrics['key_nodes_original']) & set(metrics['key_nodes_confused'])
            f.write(f"  重合节点: {list(overlap_nodes)}\n")
            f.write(f"  重合度: {metrics['overlap']}/{len(metrics['key_nodes_original'])} "
                   f"({metrics['overlap_ratio']*100:.1f}%)\n")
            
            f.write(f"\n[扰动成本]\n")
            f.write(f"  实际修改边数: {metrics['cost']}\n")
            f.write(f"  成本阈值: {metrics['cost_threshold']}\n")
            f.write(f"  是否超限: {'是' if metrics['cost_violation'] > 0 else '否'}\n")
            if metrics['cost_violation'] > 0:
                f.write(f"  超限量: {metrics['cost_violation']}\n")
            
            f.write(f"\n[文件]\n")
            f.write(f"  图像: solution_{idx:03d}.png\n")
            f.write(f"  邻接矩阵: solution_{idx:03d}_adj.txt\n")
            f.write(f"{'='*70}\n")
    
    print(f"✓ 摘要已保存: {summary_path}")
    
    # 导出每个解
    for idx, solution in enumerate(solutions, start=1):
        matrix = solution['matrix']
        key_nodes_conf = solution['metrics']['key_nodes_confused']
        
        print(f"\n处理方案 {idx}/{len(solutions)}...")
        
        # 保存图像
        img_path = os.path.join(output_dir, f"solution_{idx:03d}.png")
        DrawTopology(
            matrix=matrix, 
            critical_nodes=key_nodes_conf
        ).draw(
            show=False, 
            save_path=img_path
        )
        print(f"  ✓ 图像: {img_path}")
        
        # 保存邻接矩阵
        matrix_path = os.path.join(output_dir, f"solution_{idx:03d}_adj.txt")
        np.savetxt(matrix_path, matrix, fmt="%d")
        print(f"  ✓ 邻接矩阵: {matrix_path}")
    
    print(f"\n{'='*70}")
    print(f"批量导出完成")
    print(f"{'='*70}")
    print(f"输出目录: {output_dir}")
    print(f"导出解数量: {len(solutions)}")
    print(f"{'='*70}\n")


def interactive_mode():
    """
    交互式模式：引导用户选择操作
    """
    print(f"\n{'='*70}")
    print(f"混淆拓扑轮换动态部署系统")
    print(f"{'='*70}\n")
    
    # 选择拓扑编号
    topo_num = input("请输入拓扑编号: ").strip()
    
    # 构建文件路径
    pareto_file = f"data/{topo_num}_output_file/pareto_front.pkl"
    
    if not os.path.exists(pareto_file):
        print(f"\n[✗] 错误：找不到 Pareto 前沿文件")
        print(f"    期望路径: {pareto_file}")
        print(f"    请先运行 main.py 生成 Pareto 前沿解集\n")
        return
    
    # 选择操作模式
    print(f"\n请选择操作模式:")
    print(f"  1. 轮换部署（循环显示所有方案）")
    print(f"  2. 批量导出（一次性保存所有方案，不显示）")
    
    mode = input("\n请输入选项 (1/2): ").strip()
    
    # 筛选条件设置
    print(f"\n{'='*70}")
    print(f"设置筛选条件（直接回车表示不限制）")
    print(f"{'='*70}")
    
    max_overlap_input = input("最大重合度 (默认: 不限制): ").strip()
    max_overlap = int(max_overlap_input) if max_overlap_input else None
    
    max_cost_vio_input = input("最大成本超限量 (默认: 不限制): ").strip()
    max_cost_violation = int(max_cost_vio_input) if max_cost_vio_input else None
    
    min_sim_input = input("最小相似度 (默认: 不限制): ").strip()
    min_similarity = float(min_sim_input) if min_sim_input else None
    
    # 输出目录
    output_dir = f"data/{topo_num}_output_file/rotation_deployment/"
    
    if mode == '1':
        # 轮换部署模式
        interval_input = input("\n轮换间隔（秒，默认: 5.0）: ").strip()
        rotation_interval = float(interval_input) if interval_input else 5.0
        
        max_rot_input = input("最大轮换次数（默认: 无限）: ").strip()
        max_rotations = int(max_rot_input) if max_rot_input else None
        
        show_input = input("是否显示图像 (y/n, 默认: y): ").strip().lower()
        show_plots = show_input != 'n'
        
        save_input = input("是否保存图像 (y/n, 默认: y): ").strip().lower()
        save_images = save_input != 'n'
        
        print(f"\n开始轮换部署...")
        print(f"提示: 按 Ctrl+C 可随时中断\n")
        
        rotation_deploy(
            pareto_file=pareto_file,
            output_dir=output_dir,
            max_overlap=max_overlap,
            max_cost_violation=max_cost_violation,
            min_similarity=min_similarity,
            rotation_interval=rotation_interval,
            max_rotations=max_rotations,
            save_images=save_images,
            show_plots=show_plots
        )
    
    elif mode == '2':
        # 批量导出模式
        print(f"\n开始批量导出...")
        
        batch_export_all_solutions(
            pareto_file=pareto_file,
            output_dir=output_dir,
            max_overlap=max_overlap,
            max_cost_violation=max_cost_violation,
            min_similarity=min_similarity
        )
    
    else:
        print(f"\n[✗] 无效的选项")


if __name__ == "__main__":
    interactive_mode()
