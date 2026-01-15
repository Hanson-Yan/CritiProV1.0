import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional
from draw_topo import DrawTopology

class GraphCompressor:
    def __init__(self):
        """初始化图压缩器"""
        self.graph = None
        self.compressed_graph = None
        self.retained_nodes = set()
        self.merged_nodes = {}  # 记录被合并的节点及其所属的逻辑节点
        self.community_map = {}  # 社区检测结果
        self.node_mapping = {}  # 原始节点到压缩节点的映射
        self.supernode_counter = 0  # 超节点计数器
    
    def load_from_adjacency_matrix(self, adj_matrix: np.ndarray, edge_attributes: Optional[Dict[Tuple[int, int], Dict]] = None):
        """从邻接矩阵加载图"""
        n = adj_matrix.shape[0]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n))
        
        if edge_attributes is None:
            edge_attributes = {}
        
        undirected_attributes = {}
        for (u, v), attrs in edge_attributes.items():
            if u < v:
                undirected_attributes[(u, v)] = attrs
            else:
                undirected_attributes[(v, u)] = attrs
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] > 0:
                    attrs = undirected_attributes.get((i, j), {'weight': adj_matrix[i, j]})
                    self.graph.add_edge(i, j, **attrs)
    
    def load_adjacency_matrix_from_file(self, file_path: str):
        """从文件加载邻接矩阵"""
        adj_matrix = np.loadtxt(file_path)
        self.load_from_adjacency_matrix(adj_matrix)
        return self
    
    def load_key_nodes_from_file(self, file_path: str) -> Set[int]:
        """从文件加载关键节点（返回数字格式）"""
        with open(file_path, 'r') as f:
            key_nodes = set(map(int, f.read().splitlines()))
        return key_nodes
    
    def _identify_retained_nodes(self, key_nodes: Set[int] = None) -> Set[int]:
        """识别需要保留的节点"""
        if key_nodes is None:
            key_nodes = set()
        
        retained = set(key_nodes)
        for node, degree in self.graph.degree():
            if degree != 2 and node not in retained:
                retained.add(node)
        
        return retained
    
    def _merge_degree2_nodes(self, retained_nodes: Set[int]):
        """合并度为2的节点"""
        self.compressed_graph = nx.Graph()
        self.compressed_graph.add_nodes_from(retained_nodes)
        
        # 初始化节点映射（保留节点使用原始编号）
        self.node_mapping = {n: n for n in retained_nodes}
        processed = set(retained_nodes)
        self.merged_nodes = {}
        
        for node in list(self.graph.nodes()):
            if node in processed or self.graph.degree(node) != 2 or node in retained_nodes:
                continue
            
            neighbors = list(self.graph.neighbors(node))
            u, v = neighbors[0], neighbors[1]
            
            attrs_u = self.graph.edges[u, node]
            attrs_v = self.graph.edges[node, v]
            merged_attrs = self._merge_edge_attributes(attrs_u, attrs_v)
            
            if self.compressed_graph.has_edge(u, v):
                existing_attrs = self.compressed_graph.edges[u, v]
                self.compressed_graph.edges[u, v].update(
                    self._merge_edge_attributes(existing_attrs, merged_attrs)
                )
            else:
                self.compressed_graph.add_edge(u, v,** merged_attrs)
            
            self.merged_nodes[node] = (u, v)
            processed.add(node)
        
        # 添加保留节点之间的边
        for u, v, attrs in self.graph.edges(data=True):
            if u in retained_nodes and v in retained_nodes and not self.compressed_graph.has_edge(u, v):
                self.compressed_graph.add_edge(u, v, **attrs)
    
    def _merge_edge_attributes(self, attrs1: Dict, attrs2: Dict) -> Dict:
        """合并边属性"""
        merged = {}
        all_keys = set(attrs1.keys()).union(set(attrs2.keys()))
        
        for key in all_keys:
            val1, val2 = attrs1.get(key), attrs2.get(key)
            if val1 is None:
                merged[key] = val2
            elif val2 is None:
                merged[key] = val1
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                merged[key] = val1 + val2
            elif isinstance(val1, list) and isinstance(val2, list):
                merged[key] = val1 + val2
            else:
                merged[key] = val1
        return merged
    
    def _get_new_supernode_id(self) -> int:
        """获取新的超节点编号（从原始节点数+1开始）"""
        original_node_count = len(self.graph.nodes)
        self.supernode_counter += 1
        return original_node_count + self.supernode_counter
    
    def _detect_communities(self, method: str = 'louvain'):
        """检测社区结构"""
        graph_to_use = self.graph if self.compressed_graph is None else self.compressed_graph
        if method == 'louvain':
            communities = louvain_communities(graph_to_use, weight='weight')
            for i, comm in enumerate(communities):
                for node in comm:
                    self.community_map[node] = i
        else:
            raise ValueError(f"不支持的社区检测方法: {method}")
    
    def _community_aggregation(self, key_nodes: Set[int] = None):
        """基于社区进行聚合"""
        if not self.community_map:
            self._detect_communities()
        
        if key_nodes is None:
            key_nodes = set()
        
        community_graph = nx.Graph()
        original_node_count = len(self.graph.nodes)
        
        # 添加关键节点（保留原始编号）
        for node in key_nodes:
            if node in self.compressed_graph.nodes:
                community_graph.add_node(node)
        
        # 添加社区作为超节点（新编号区间）
        communities = set(self.community_map.values())
        supernode_mapping = {}  # 社区ID到超节点ID的映射
        
        for comm_id in communities:
            comm_nodes = [n for n in self.community_map if self.community_map[n] == comm_id]
            has_key_node = any(node in key_nodes for node in comm_nodes)
            
            if not has_key_node:
                supernode_id = self._get_new_supernode_id()
                community_graph.add_node(supernode_id)
                supernode_mapping[comm_id] = supernode_id
                
                # 记录超节点包含的原始节点
                for node in comm_nodes:
                    self.node_mapping[node] = supernode_id
        
        # 添加社区间的边
        for u, v, attrs in self.compressed_graph.edges(data=True):
            # 处理u节点
            if u in key_nodes:
                u_node = u
            else:
                u_comm = self.community_map.get(u, None)
                if u_comm is not None and u_comm in supernode_mapping:
                    u_node = supernode_mapping[u_comm]
                else:
                    continue  # 异常情况，跳过
            
            # 处理v节点
            if v in key_nodes:
                v_node = v
            else:
                v_comm = self.community_map.get(v, None)
                if v_comm is not None and v_comm in supernode_mapping:
                    v_node = supernode_mapping[v_comm]
                else:
                    continue  # 异常情况，跳过
            
            if u_node != v_node:
                if community_graph.has_edge(u_node, v_node):
                    existing_attrs = community_graph.edges[u_node, v_node]
                    community_graph.edges[u_node, v_node].update(
                        self._merge_edge_attributes(existing_attrs, attrs)
                    )
                else:
                    community_graph.add_edge(u_node, v_node, **attrs)
        
        self.compressed_graph = community_graph
    
    def compress(self, key_nodes_file: Optional[str] = None, community_aggregation: bool = False, 
                 community_method: str = 'louvain') -> nx.Graph:
        """执行图压缩"""
        key_nodes = set()
        if key_nodes_file:
            key_nodes = self.load_key_nodes_from_file(key_nodes_file)
        
        self.retained_nodes = self._identify_retained_nodes(key_nodes)
        self._merge_degree2_nodes(self.retained_nodes)
        
        if community_aggregation:
            self._detect_communities(method=community_method)
            self._community_aggregation(key_nodes)
        
        return self.compressed_graph
    
    def save_mapping_table(self, file_path: str):
        """保存节点映射表"""
        mapping = {}
        for original_node, compressed_node in self.node_mapping.items():
            if compressed_node not in mapping:
                mapping[compressed_node] = []
            mapping[compressed_node].append(original_node)
        
        with open(file_path, 'w') as f:
            for compressed_node, original_nodes in mapping.items():
                f.write(f"{compressed_node}: {','.join(map(str, original_nodes))}\n")
    
    def save_compressed_graph(self, file_path: str, format: str = 'adjacency_matrix'):
        """保存压缩后的图"""
        if self.compressed_graph is None:
            raise ValueError("请先执行压缩操作")
        
        if format == 'adjacency_matrix':
            # 转换为邻接矩阵时，确保节点编号连续
            nodes = sorted(self.compressed_graph.nodes)
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)
            adj_matrix = np.zeros((n, n))
            
            for u, v, data in self.compressed_graph.edges(data=True):
                i, j = node_to_idx[u], node_to_idx[v]
                adj_matrix[i, j] = data.get('weight', 1)
                adj_matrix[j, i] = adj_matrix[i, j]  # 无向图对称
                
            np.savetxt(file_path, adj_matrix, fmt='%d')
        elif format == 'edgelist':
            nx.write_edgelist(self.compressed_graph, file_path, data=True)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
    
    def get_compression_stats(self) -> Dict:
        """获取压缩统计信息"""
        if self.graph is None or self.compressed_graph is None:
            return {}
        
        original_nodes = self.graph.number_of_nodes()
        original_edges = self.graph.number_of_edges()
        compressed_nodes = self.compressed_graph.number_of_nodes()
        compressed_edges = self.compressed_graph.number_of_edges()
        
        return {
            'original_nodes': original_nodes,
            'original_edges': original_edges,
            'compressed_nodes': compressed_nodes,
            'compressed_edges': compressed_edges,
            'node_reduction': 1 - compressed_nodes / original_nodes,
            'edge_reduction': 1 - compressed_edges / original_edges,
            'merged_nodes_count': len(self.merged_nodes),
            'supernode_count': self.supernode_counter
        }


# 使用示例
if __name__ == "__main__":
    # 生成包含度≠2节点的测试图
    def generate_test_adjacency(n_nodes=100):
        G = nx.barabasi_albert_graph(n_nodes, 2)  # 无标度图，有很多度>2的节点
        return nx.to_numpy_array(G)
    
    np.random.seed(42)
    size = 500  # 小规模测试
    adj_matrix = generate_test_adjacency(size)
    
    # 添加边属性
    edge_attributes = {}
    for i in range(size):
        for j in range(i + 1, size):
            if adj_matrix[i, j] > 0:
                edge_attributes[(i, j)] = {
                    'weight': np.random.randint(1, 10),
                    'length': np.random.uniform(0.5, 10.0)
                }
    
    # 初始化压缩器
    compressor = GraphCompressor()
    compressor.load_from_adjacency_matrix(adj_matrix, edge_attributes)
    
    # 定义关键节点（数字格式）
    key_nodes = {0, 5, 10, 15}
    with open('key_nodes.txt', 'w') as f:
        for node in key_nodes:
            f.write(f"{node}\n")
    
    # 绘制原始拓扑图
    print("绘制原始拓扑图...")
    # 原始关键节点转换为sX格式（与DrawTopology兼容）
    original_key_nodes = [f"s{node}" for node in key_nodes]
    DrawTopology(matrix=adj_matrix, critical_nodes=original_key_nodes).draw(show=True)
    
    # 执行压缩（不使用社区聚合）
    compressed_graph = compressor.compress(
        key_nodes_file='key_nodes.txt',
        community_aggregation=False
    )
    
    # 打印压缩统计
    stats = compressor.get_compression_stats()
    print("\n压缩统计（不使用社区聚合）:")
    for key, value in stats.items():
        if key in ['node_reduction', 'edge_reduction']:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    
    # 保存映射表和压缩图
    compressor.save_mapping_table('node_mapping.txt')
    compressor.save_compressed_graph('compressed_no_community.txt', format='adjacency_matrix')
    
    # 加载压缩后的矩阵
    matrix = np.loadtxt('compressed_no_community.txt')
    
    # 关键节点转换为sX格式（与DrawTopology兼容）
    # 1. 获取压缩图的节点映射关系（原始节点 -> 压缩后节点）
    compressed_to_original = {}
    for orig, comp in compressor.node_mapping.items():
        if comp not in compressed_to_original:
            compressed_to_original[comp] = []
        compressed_to_original[comp].append(orig)
    
    # 2. 找出在压缩图中存在的关键节点（原始编号）
    valid_original_keys = [n for n in key_nodes if n in compressor.node_mapping]
    
    # 3. 获取这些关键节点在压缩图中的索引（邻接矩阵的行/列索引）
    # 压缩图的节点按顺序对应邻接矩阵的0,1,2...索引
    compressed_nodes = sorted(compressed_graph.nodes)
    key_node_indices = [compressed_nodes.index(n) for n in valid_original_keys if n in compressed_nodes]
    
    # 4. 转换为sX格式（与DrawTopology要求一致）
    valid_key_nodes = [f"s{i}" for i in key_node_indices]
    
    # 绘制压缩后拓扑图（不使用社区聚合）
    print("绘制压缩后拓扑图（不使用社区聚合）...")
    DrawTopology(matrix=matrix, critical_nodes=valid_key_nodes).draw(show=True)
    
    # 执行社区聚合压缩
    compressor_community = GraphCompressor()
    compressor_community.load_from_adjacency_matrix(adj_matrix, edge_attributes)
    compressed_graph_community = compressor_community.compress(
        key_nodes_file='key_nodes.txt',
        community_aggregation=True,
        community_method='louvain'
    )
    
    # 打印社区聚合后的统计
    stats_community = compressor_community.get_compression_stats()
    print("\n压缩统计（使用社区聚合）:")
    for key, value in stats_community.items():
        if key in ['node_reduction', 'edge_reduction']:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    
    # 保存社区聚合后的映射表和图
    compressor_community.save_mapping_table('node_mapping_community.txt')
    compressor_community.save_compressed_graph('compressed_with_community.txt', format='adjacency_matrix')
    
    # 加载社区聚合后的矩阵并处理节点格式
    matrix_community = np.loadtxt('compressed_with_community.txt')
    community_nodes = sorted(compressed_graph_community.nodes)
    
    # 社区聚合后的关键节点转换为sX格式
    valid_original_keys_comm = [n for n in key_nodes if n in compressor_community.node_mapping]
    key_node_indices_comm = [community_nodes.index(n) for n in valid_original_keys_comm if n in community_nodes]
    valid_key_nodes_comm = [f"s{i}" for i in key_node_indices_comm]
    
    # 绘制压缩后拓扑图（使用社区聚合）
    print("绘制压缩后拓扑图（使用社区聚合）...")
    DrawTopology(matrix=matrix_community, critical_nodes=valid_key_nodes_comm).draw(show=True)
    
    # 打印节点映射表示例
    print("\n节点映射表示例:")
    with open('node_mapping_community.txt', 'r') as f:
        for line in f.readlines()[:3]:  # 打印前3行
            print(line.strip())

