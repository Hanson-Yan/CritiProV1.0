import numpy as np
import networkx as nx
import math
from collections import defaultdict

def compute_static_metrics(adj_matrix):
    """
    计算网络拓扑的静态指标。
    """
    G = nx.Graph(adj_matrix)
    
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = nx.degree_centrality(G)
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # 计算最小割中心性
    node_edge_betweenness = {node: 0 for node in G.nodes}
    for (u, v), betweenness_value in edge_betweenness.items():
        node_edge_betweenness[u] += betweenness_value
        node_edge_betweenness[v] += betweenness_value
    
    metrics = {
        node: {
            "betweenness_centrality": betweenness[node],
            "closeness_centrality": closeness[node],
            "degree_centrality": degree[node],
            "min_cut_centrality": node_edge_betweenness[node]
        }
        for node in G.nodes
    }
    return metrics

def normalize_data(data):
    """
    标准化数据
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
    
    epsilon = 1e-12  # 避免除零
    p_ij = normalized_data / normalized_data.sum(axis=0)
    e_j = -np.sum((p_ij * np.log(p_ij + epsilon)), axis=0) / len(data)
    
    weights = (1 - e_j) / (1 - e_j).sum()
    return dict(zip(metrics, weights))

def topsis_method(data, weights):
    """
    使用TOPSIS方法计算每个节点的得分。
    """
    metrics = list(next(iter(data.values())).keys())
    normalized_data = np.array([[data[node][metric] for metric in metrics] for node in data])
    
    weighted_data = normalized_data * np.array(list(weights.values()))
    
    ideal_best = weighted_data.max(axis=0)
    ideal_worst = weighted_data.min(axis=0)
    
    distance_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    
    scores = distance_worst / (distance_best + distance_worst)
    return dict(zip(data.keys(), scores))

def identify_key_nodes(adj_matrix, top_k_ratio=0.2):
    """
    识别关键节点
    """
    node_metrics = compute_static_metrics(adj_matrix)
    normalized_metrics = normalize_data(node_metrics)
    weights = entropy_weight_method(normalized_metrics)
    scores = topsis_method(normalized_metrics, weights)
    
    k = math.ceil(len(node_metrics) * top_k_ratio)
    key_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return key_nodes

def key_node_identify(test_matrix_confuse_txt,topo_num):
    # 示例邻接矩阵
    data_dir = f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/"
    result_dir = f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/"

    topo_matrix = np.loadtxt(data_dir + test_matrix_confuse_txt)
    result_file = f"{topo_num}_critinode_of_"+test_matrix_confuse_txt.split("_")[0] 
    result_path = result_dir + result_file + ".txt"

    # 识别关键节点
    key_nodes = identify_key_nodes(topo_matrix)
    print(f"{test_matrix_confuse_txt.split("_")[0]}识别的关键节点如下：")
    result_node = []
    for node, score in key_nodes:
        print(f"节点 {node}: 得分 = {score:.4f}")
        result_node.append(node)
    np.savetxt(result_path, result_node, fmt="%d")
    print(f"{test_matrix_confuse_txt.split("_")[0]} 的结果写入\n{result_path}")

def key_node_critipro(topo_num):
    import os
    
    critipro_critinode=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/data/{topo_num}_output_file/critinode_critipro.txt"
    citinode_of_critipro = f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_critipro.txt"
    original_critinode=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/critical_nodes.txt"
    original_of_critipro = f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_original.txt"

    def copy_file(source_file, target_file):
        try:
            # 打开源文件用于读取
            with open(source_file, 'r') as src:
                content = src.read()  # 读取源文件内容

            # 打开目标文件用于写入（覆盖）
            with open(target_file, 'w') as tgt:
                tgt.write(content)  # 将源文件内容写入目标文件

            print(f"成功将 {source_file} 的内容拷贝到 {target_file}")
        except FileNotFoundError:
            print(f"错误：文件 {source_file} 未找到")
        except Exception as e:
            print(f"发生错误：{e}")
    
    if not os.path.exists(critipro_critinode):
        print(f"{critipro_critinode} is not exist")
        exit(1)
    # if not os.path.exists(citinode_of_critipro):
    #     print(f"{citinode_of_critipro} is not exist")
    #     exit(1)   
    copy_file(critipro_critinode,citinode_of_critipro)
    copy_file(original_critinode,original_of_critipro)

def key_node_static(topo_num):
    key_node_critipro(topo_num)
    key_node_identify(test_matrix_confuse_txt = f"antitomo_{topo_num}_confuse_topo.txt",topo_num=topo_num)
    key_node_identify(test_matrix_confuse_txt = f"proto_{topo_num}_confuse_topo.txt",topo_num=topo_num)

# if __name__ =="__main__":
#     key_node_static(topo_num)
    