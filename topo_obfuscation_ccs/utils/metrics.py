import networkx as nx
import numpy as np
from typing import Dict, Optional
from collections import deque
import random

def approx_betweenness(G: nx.Graph, k: Optional[int] = 10, seed: Optional[int] = None) -> Dict:
    """
    基于 Brandes 单源最短路径的采样近似 betweenness centrality。
    - G: NetworkX 图
    - k: 采样源点数量（若 k >= n 或为 None 则退化为精确计算）
    返回每个节点的近似 betweenness 值（未严格归一化，但可用于排序/比较）。
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    if k is None or k >= n:
        return nx.betweenness_centrality(G)


    rng = random.Random(seed)
    sources = rng.sample(nodes, min(k, n))


    CB = dict.fromkeys(nodes, 0.0)


    for s in sources:
        # Brandes 单源流程（无权图）
        S = []
        P = {v: [] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, -1)
        dist[s] = 0


        Q = deque([s])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in G.neighbors(v):
                if dist[w] < 0:
                    Q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)


        delta = dict.fromkeys(nodes, 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] != 0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                else:
                    delta_v = 0.0
                delta[v] += delta_v
            if w != s:
                CB[w] += delta[w]


    # 对采样结果做放大（粗略），使得不同 k 值下的可比较性更好
    factor = float(n - 1) / max(1, len(sources))
    for v in CB:
        CB[v] = CB[v] * factor
    return CB



class VirtualNodeMetrics:
    def __init__(self, topo_matrix: np.ndarray, original_dynamic_metrics: Dict):
        self.topo_matrix = topo_matrix
        self.original_dynamic_metrics = original_dynamic_metrics
        self.graph = self._create_graph()
        self.static_metrics = self._calculate_static_metrics()
        self.dynamic_metrics = self._calculate_dynamic_metrics()

    def _create_graph(self):
        """根据拓扑矩阵创建 NetworkX 图"""
        return nx.from_numpy_array(self.topo_matrix)

    def _calculate_static_metrics(self):
        """计算混淆拓扑的静态指标"""
        # betweenness = nx.betweenness_centrality(self.graph)
        betweenness = approx_betweenness(self.graph)
        closeness = nx.closeness_centrality(self.graph)
        degree = nx.degree_centrality(self.graph)
        edge_betweenness = nx.edge_betweenness_centrality(self.graph)

        # 计算节点的最小割中心性
        node_edge_betweenness = {node: 0 for node in self.graph.nodes}
        for (u, v), betweenness_value in edge_betweenness.items():
            node_edge_betweenness[u] += betweenness_value
            node_edge_betweenness[v] += betweenness_value

        static_metrics = {
            node: {
                'betweenness_centrality': betweenness[node],
                'closeness_centrality': closeness[node],
                'degree_centrality': degree[node],
                'node_edge_betweenness': node_edge_betweenness[node]
            }
            for node in self.graph.nodes
        }

        return static_metrics

    def _calculate_dynamic_metrics(self):
        """计算混淆拓扑的虚拟动态指标"""
        dynamic_metrics = {}

        # 计算混淆拓扑的静态指标
        degree_centrality = nx.degree_centrality(self.graph)

        epsilon = 1e-6  # 避免除 0

        for node in self.graph.nodes:
            node_name = f"s{node}"  # 确保匹配原始指标字典的键格式
            
            if node_name in self.original_dynamic_metrics:
                original_data = self.original_dynamic_metrics[node_name]

                # 获取原始拓扑的 `degree_centrality`
                original_degree = original_data['degree_centrality']

                # 获取混淆拓扑的 `degree_centrality`
                new_degree = degree_centrality.get(node, 0)

                # 计算调整因子
                adjust_factor = (new_degree + epsilon) / (max(original_degree, epsilon))

                # 计算新的动态指标
                new_traffic_load = original_data['aggregate_traffic_load'] * adjust_factor
                new_utilized_bw_ratio = original_data['utilized_bandwidth_ratio'] * adjust_factor
                new_available_bw = original_data['available_link_bandwidth'] * (2 - adjust_factor)
                new_link_delay = original_data['link_delay']  # 不变

                # 存储计算后的虚拟动态指标
                dynamic_metrics[node_name] = {
                    'aggregate_traffic_load': new_traffic_load,
                    'utilized_bandwidth_ratio': new_utilized_bw_ratio,
                    'available_link_bandwidth': max(new_available_bw, 0),  # 确保带宽不为负
                    'link_delay': new_link_delay
                }

        return dynamic_metrics

    def get_metrics(self):
        """返回完整的混淆拓扑指标（静态 + 动态），格式化为字典"""
        combined_metrics = {}
        for node in self.graph.nodes:
            node_name = f"s{node}"  
            combined_metrics[node_name] = {
                **self.static_metrics.get(node, {}),  
                **self.dynamic_metrics.get(node_name, {})  
            }
        return combined_metrics
