
# similarity_value_verify.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

utils_dir = "/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/utils"
sys.path.append(utils_dir)
from similarity import graph_similarity

def normalize_matrix(matrix):
    return (matrix > 0).astype(int)

def is_connected(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    return nx.is_connected(G)

def find_perturbed_by_similarity_range(original, simi_min, simi_max, max_trials=3000):
    """
    使用 portrait 方法计算相似度
    """
    n = original.shape[0]
    original = normalize_matrix(original)
    edge_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(max_trials):
        perturbed = deepcopy(original)
        num_changes = np.random.randint(1, len(edge_list) // 2)
        flip_indices = np.random.choice(len(edge_list), size=num_changes, replace=False)
        
        for idx in flip_indices:
            u, v = edge_list[idx]
            perturbed[u][v] = 1 - perturbed[u][v]
            perturbed[v][u] = perturbed[u][v]
        
        if is_connected(perturbed):
            # 调用 similarity.py 中的 portrait 方法
            simi = graph_similarity(original, perturbed, method='portrait')
            
            if trial % 100 == 0:  # 每100次迭代打印一次进度
                print(f"Trial {trial}: simi={simi:.4f}, target=[{simi_min}, {simi_max}]")
            
            if simi_min < simi <= simi_max:
                print(f"✓ Found at trial {trial}: simi={simi:.4f}")
                return perturbed, simi
    
    print(f"✗ Failed to find topology in range [{simi_min}, {simi_max}] after {max_trials} trials")
    return original, 1.0  # fallback


def draw_topologies_by_similarity_ranges(original_matrix, ranges, topo_num):
    results = []
    titles = []
    
    for simi_min, simi_max in ranges:
        print(f"\n{'='*50}")
        print(f"Searching for topology in range [{simi_min}, {simi_max}]...")
        
        if simi_max == 1.0 and simi_min == 1.0:  # 保留原图
            mat = original_matrix
            simi = 1.0
            print("Using original topology")
        else:
            mat, simi = find_perturbed_by_similarity_range(original_matrix, simi_min, simi_max)
        
        results.append(mat)
        
        if simi_max == 1.0 and simi_min == 1.0:
            titles.append(f"Original Topology")
        else:
            titles.append(f"{simi_min:.1f}~{simi_max:.1f}\nsimi≈{simi:.4f}")

    # 绘图部分
    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    
    # 处理单个子图的情况
    if len(results) == 1:
        axes = [axes]
    
    for i, (matrix, ax) in enumerate(zip(results, axes)):
        G = nx.Graph()
        n = matrix.shape[0]
        G.add_nodes_from([f"s{i}" for i in range(n)])
        
        for u in range(n):
            for v in range(u + 1, n):
                if matrix[u][v] > 0:
                    G.add_edge(f"s{u}", f"s{v}")
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen', node_size=200)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.set_title(titles[i], fontsize=20)
        ax.text(0.5, -0.15, f'{chr(65 + i)}', ha='center', va='center', 
                transform=ax.transAxes, fontweight='bold', fontsize=20)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = f"/home/retr0/Project/TopologyObfu/Experiment/simi_{topo_num}.png"
    plt.savefig(output_path, format='png', dpi=600)
    print(f"\n{'='*50}")
    print(f"Saved figure to: {output_path}")
    plt.show()


# ---------- 示例入口 ----------
if __name__ == "__main__":
    topo_num = input("Please input topo_num: ")
    
    # 加载原始邻接矩阵
    # path = f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{topo_num}.txt"
    path = f"/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/{topo_num}.txt"
    print(f"Loading topology from: {path}")
    original_adj = np.loadtxt(path)
    print(f"Topology size: {original_adj.shape[0]} nodes")
    
    # 新的相似度区间：0~0.6, 0.6~0.9, 0.9~1
    simi_ranges = [
        (0.0, 0.6),
        (0.6, 0.9),
        (0.9, 0.999),
        (1.0, 1.0)  # 原图
    ]
    
    print(f"\nTarget similarity ranges: {simi_ranges}")
    draw_topologies_by_similarity_ranges(original_adj, simi_ranges, topo_num)
