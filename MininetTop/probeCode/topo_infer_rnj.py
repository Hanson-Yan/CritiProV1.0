# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from copy import deepcopy

# class Node:
#     def __init__(self, label):
#         self.label = label          # 原始标签
#         self.display_label = None   # 显示标签
#         self.parent = None
#         self.children = []

# def RNJ_algorithm(distance_matrix):
#     n = len(distance_matrix)
#     original_nodes = {str(i): Node(str(i)) for i in range(n)}
#     leaves = list(original_nodes.values())
    
#     graph = nx.DiGraph()
#     # 统一使用s前缀编号
#     for node in original_nodes.values():
#         node.display_label = f"s{node.label}"   # 所有节点用s开头
#         graph.add_node(node.display_label)
    
#     def compute_shared_length(node_i, node_j):
#         labels_i = node_i.label.split(':') if '(' in node_i.label else [node_i.label]
#         labels_j = node_j.label.split(':') if '(' in node_j.label else [node_j.label]
        
#         total = 0.0
#         count = 0
#         for li in labels_i:
#             for lj in labels_j:
#                 if li in original_nodes and lj in original_nodes:
#                     total += distance_matrix[int(li)][int(lj)]
#                     count += 1
#         return total / count if count > 0 else 0.0

#     def build_tree(working_leaves):
#         if len(working_leaves) == 1:
#             return working_leaves[0]
        
#         max_shared = -np.inf
#         best_pair = (0, 1)
#         for i in range(len(working_leaves)):
#             for j in range(i+1, len(working_leaves)):
#                 current = compute_shared_length(working_leaves[i], working_leaves[j])
#                 if current > max_shared:
#                     max_shared, best_pair = current, (i, j)
        
#         i, j = best_pair
#         node_i, node_j = working_leaves[i], working_leaves[j]
        
#         # 创建中间节点
#         parent_label = f"({node_i.label}:{node_j.label})"
#         parent = Node(parent_label)
#         parent.display_label = f"s{len(graph.nodes)}"  # 持续使用s编号
#         parent.children = [node_i, node_j]
#         node_i.parent = node_j.parent = parent
        
#         graph.add_node(parent.display_label)
#         graph.add_edge(parent.display_label, node_i.display_label)
#         graph.add_edge(parent.display_label, node_j.display_label)
        
#         new_leaves = [node for idx, node in enumerate(working_leaves) if idx not in {i, j}]
#         new_leaves.append(parent)
        
#         return build_tree(new_leaves)
    
#     root = build_tree(leaves)  # 直接返回原始根节点
#     return root, graph

# def print_tree(node, indent=""):
#     print(indent + node.display_label)
#     for child in node.children:
#         print_tree(child, indent + "  ")

# def visualize_tree(graph):
#     plt.figure(figsize=(12, 8))
#     pos = nx.spring_layout(graph, seed=42)
    
#     # 根据节点类型动态着色
#     node_colors = []
#     for node in graph.nodes:
#         # 通过出度判断节点类型（中间节点有出边）
#         if graph.out_degree(node) > 0:
#             node_colors.append('#FF6666')  # 中间节点红色
#         else:
#             node_colors.append('#6666FF')  # 叶子节点蓝色
    
#     nx.draw(graph, pos,
#            node_color=node_colors,
#            node_size=800,
#            with_labels=True,
#            edge_color='grey',
#            arrowsize=20,
#            font_size=10)
    
#     plt.title("Unified s-labeled Network Topology")
#     plt.show()

# # 数据加载（保持不变）
# data_path = "/home/retr0/Project/TopologyObfu/MininetTop/probeCode/host_result/"
# delay_matrix = np.loadtxt(data_path + "delay_matrix.txt")

# n = len(delay_matrix)
# distance_matrix = np.zeros((n, n))
# for i in range(n):
#     for j in range(i, n):
#         distance_matrix[i][j] = delay_matrix[i][j]
#         distance_matrix[j][i] = delay_matrix[i][j]

# # 运行算法
# root, graph = RNJ_algorithm(distance_matrix)

# print("Unified Tree Structure:")
# print_tree(root)

# # 可视化
# visualize_tree(graph)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

class Node:
    def __init__(self, label):
        self.label = label          # 原始标签
        self.display_label = None   # 显示标签
        self.parent = None
        self.children = []

def RNJ_algorithm(distance_matrix):
    n = len(distance_matrix)
    original_nodes = {str(i): Node(str(i)) for i in range(n)}
    leaves = list(original_nodes.values())
    
    graph = nx.Graph()  # 使用无向图
    # 统一使用s前缀编号
    for node in original_nodes.values():
        node.display_label = f"s{node.label}"   # 所有节点用s开头
        graph.add_node(node.display_label)
    
    def compute_shared_length(node_i, node_j):
        labels_i = node_i.label.split(':') if '(' in node_i.label else [node_i.label]
        labels_j = node_j.label.split(':') if '(' in node_j.label else [node_j.label]
        
        total = 0.0
        count = 0
        for li in labels_i:
            for lj in labels_j:
                if li in original_nodes and lj in original_nodes:
                    total += distance_matrix[int(li)][int(lj)]
                    count += 1
        return total / count if count > 0 else 0.0

    def build_tree(working_leaves):
        if len(working_leaves) == 1:
            return working_leaves[0]
        
        max_shared = -np.inf
        best_pair = (0, 1)
        for i in range(len(working_leaves)):
            for j in range(i+1, len(working_leaves)):
                current = compute_shared_length(working_leaves[i], working_leaves[j])
                if current > max_shared:
                    max_shared, best_pair = current, (i, j)
        
        i, j = best_pair
        node_i, node_j = working_leaves[i], working_leaves[j]
        
        # 创建中间节点
        parent_label = f"({node_i.label}:{node_j.label})"
        parent = Node(parent_label)
        parent.display_label = f"s{len(graph.nodes)}"  # 持续使用s编号
        parent.children = [node_i, node_j]
        node_i.parent = node_j.parent = parent
        
        graph.add_node(parent.display_label)
        graph.add_edge(parent.display_label, node_i.display_label)  # 添加无向边
        graph.add_edge(parent.display_label, node_j.display_label)  # 添加无向边
        
        new_leaves = [node for idx, node in enumerate(working_leaves) if idx not in {i, j}]
        new_leaves.append(parent)
        
        return build_tree(new_leaves)
    
    root = build_tree(leaves)  # 直接返回原始根节点
    return root, graph

def print_tree(node, indent=""):
    print(indent + node.display_label)
    for child in node.children:
        print_tree(child, indent + "  ")

def visualize_tree(graph, save_path=None):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    node_colors = ['#6666FF'] * len(graph.nodes)  # 统一颜色
    nx.draw(graph, pos,
           node_color=node_colors,
           node_size=800,
           with_labels=True,
           edge_color='grey',
           width=2.0,  # 控制边的宽度
           font_size=10)
    if save_path:
        plt.savefig(save_path)
        print(f"fig have saved in\n{save_path}")
    plt.title("Unified s-labeled Network Topology (Undirected)")
    plt.show()

# 数据加载（保持不变）
topo_num = input("please input topo_num:")
data_path = "/home/retr0/Project/TopologyObfu/MininetTop/probeCode/host_result/"
delay_matrix = np.loadtxt(data_path +topo_num+ "_delay_matrix.txt")
save_path = "/home/retr0/Project/TopologyObfu/MininetTop/probeCode/topo_infer_data/original_topo_infer.png"

n = len(delay_matrix)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        distance_matrix[i][j] = delay_matrix[i][j]
        distance_matrix[j][i] = delay_matrix[i][j]

# 运行算法
root, graph = RNJ_algorithm(distance_matrix)

print(f"data from:\n{data_path + "delay_matrix.txt"}")
print("Unified Tree Structure:")
print_tree(root)

# 可视化
visualize_tree(graph,save_path)
print(f"originial topo infer result have saved\n{save_path}")