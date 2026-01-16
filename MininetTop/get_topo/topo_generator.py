import networkx as nx
import numpy as np
import random

def generate_random_topology(graph_type="ER", n=10, p=0.2, m=2, k=4, beta=0.1, save_path=None, seed=None):
    """
    生成随机拓扑并转为邻接矩阵，确保连通
    :param graph_type: 拓扑类型 ("ER", "BA", "WS")
    :param n: 节点数
    :param p: ER 模型连边概率 (建议 >= log(n)/n，默认 0.2)
    :param m: BA 模型每个新节点连边数 (需满足 1 <= m < n)
    :param k: WS 模型每个节点初始邻居数 (需满足 2 <= k < n)
    :param beta: WS 模型重连概率
    :param save_path: 邻接矩阵保存路径 (txt)
    :param seed: 随机种子
    :return: 邻接矩阵 (numpy array)
    """
    if seed is not None:
        np.random.seed(seed)

    if graph_type == "ER":
        # 确保 p 足够大以避免孤立点
        p_min = np.log(n) / n
        if p < p_min:
            print(f"[警告] p={p} 过小，可能不连通，建议使用 p >= {p_min:.3f}")
        G = nx.erdos_renyi_graph(n, p, seed=seed)

    elif graph_type == "BA":
        if m < 1 or m >= n:
            raise ValueError("BA 模型参数错误: 需要满足 1 <= m < n")
        G = nx.barabasi_albert_graph(n, m, seed=seed)

    elif graph_type == "WS":
        if k < 2 or k >= n:
            raise ValueError("WS 模型参数错误: 需要满足 2 <= k < n")
        G = nx.watts_strogatz_graph(n, k, beta, seed=seed)

    else:
        raise ValueError("Unsupported graph type! Choose from ER, BA, WS.")

    # 如果图不连通，直接取最大连通分量（保证无孤立点）
    if not nx.is_connected(G):
        print("[提示] 生成的图不连通，已取最大连通分量")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    adj_matrix = nx.to_numpy_array(G, dtype=int)

    if save_path:
        np.savetxt(save_path, adj_matrix, fmt="%d")
        print(f"拓扑已保存到 {save_path}")

    return adj_matrix

def parse_topology_zoo(file_path, save_path=None):
    """
    解析 Topology Zoo 的 .graphml / .gml 文件，转成邻接矩阵
    :param file_path: 输入文件路径 (.graphml 或 .gml)
    :param save_path: 输出邻接矩阵保存路径 (txt)
    :return: 邻接矩阵 (numpy array)
    """
    if file_path.endswith(".graphml"):
        G = nx.read_graphml(file_path)
    elif file_path.endswith(".gml"):
        G = nx.read_gml(file_path)
    else:
        raise ValueError("File format not supported! Please use .graphml or .gml.")

    # 确保节点按编号顺序排序
    G = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
    adj_matrix = nx.to_numpy_array(G, dtype=int)

    if save_path:
        np.savetxt(save_path, adj_matrix, fmt="%d")
        print(f"拓扑已保存到 {save_path}")

    return adj_matrix


def get_topology(input_type="random", **kwargs):
    """
    统一拓扑输入接口
    :param input_type: "random" 或 "zoo"
    :param kwargs: 参数，取决于类型
    :return: 邻接矩阵
    """
    if input_type == "random":
        return generate_random_topology(**kwargs)
    elif input_type == "zoo":
        return parse_topology_zoo(**kwargs)
    else:
        raise ValueError("Unsupported input_type! Choose from 'random' or 'zoo'.")

def assign_hosts_to_switches(adj_matrix, host_num=None, strategy="random", save_path=None):
    """
    为拓扑中的交换机分配主机
    :param adj_matrix: 邻接矩阵 (numpy array)
    :param host_num: 主机数 (默认按比例分配)
    :param strategy: 分配策略 ("random" 或 "degree")
    :param save_path: 输出文件路径
    :return: (host_num, switch_num, connect_switch_order)
    """
    switch_num = adj_matrix.shape[0]

    # 默认主机数 = 节点数的 2/3 （可调）
    if host_num is None:
        host_num = max(1, switch_num // 2)

    G = nx.from_numpy_array(adj_matrix)

    if strategy == "random":
        connect_switch_order = random.sample(range(switch_num), host_num)
    elif strategy == "degree":
        # 度数小的节点更可能是边缘节点
        degree_sorted = sorted(G.degree, key=lambda x: x[1])  
        connect_switch_order = [node for node, _ in degree_sorted[:host_num]]
    else:
        raise ValueError("Unsupported strategy! Choose 'random' or 'degree'.")

    # 保存结果
    if save_path:
        with open(save_path, "w") as f:
            f.write(f"host_num: {host_num}\n")
            f.write(f"switch_num: {switch_num}\n")
            f.write("connect_switch_order: " + ", ".join(map(str, connect_switch_order)) + "\n")
        print(f"主机分配信息已保存到 {save_path}")

    return host_num, switch_num, connect_switch_order

"""
# 随机拓扑
adj1 = get_topology(input_type="random", graph_type="ER", n=20, p=0.2, save_path="er20.txt")

# Topology Zoo
adj2 = get_topology(input_type="zoo", file_path="Abilene.graphml", save_path="abilene.txt")


# 生成一个 10 节点 ER 拓扑
adj = generate_random_topology("ER", n=10, p=0.3, save_path="er10.txt")

# 给拓扑分配 6 个主机，随机挂载
assign_hosts_to_switches(adj, host_num=6, strategy="random", save_path="er10_info.txt")

# 给拓扑分配主机，挂到度数最小的交换机上
assign_hosts_to_switches(adj, host_num=6, strategy="degree", save_path="er10_info_degree.txt")
"""
def main():
    topo_num = "topo_6"
    topo_matrix_path = f"/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/"
    zoo_topo_name = "Litnet.gml"

    num=int(input("host num:"))
    adj1 = get_topology(input_type="random", graph_type="WS", n=num, p=0.2, save_path=f"{topo_matrix_path}{topo_num}.txt")
    # adj2 = get_topology(input_type="zoo", file_path=f"./dataset/{zoo_topo_name}", save_path=f"{topo_matrix_path}{topo_num}.txt")
    # assign_hosts_to_switches(adj1, host_num=None, strategy="random", save_path=f"{topo_matrix_path}{topo_num}_info.txt")
    assign_hosts_to_switches(adj1, host_num=None, strategy="random", save_path=f"{topo_matrix_path}{topo_num}_info.txt")
    

if __name__ == "__main__":
    main()




