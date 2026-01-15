# encoder.py

import numpy as np
import networkx as nx
from typing import List, Tuple

class SparseEdgeEncoder:
    def __init__(self, n_nodes: int, original_matrix: np.ndarray, b_hop: int = 2):
        self.n = n_nodes
        self.original_matrix = original_matrix
        self.original_graph = nx.from_numpy_array(original_matrix)
        self.edge_list = self._generate_modifiable_edge_set(b_hop)

        self.n_modifiable_edges = len(self.edge_list)
        

    def _generate_modifiable_edge_set(self, b_hop: int) -> List[Tuple[int, int]]:
        """根据 b-hop 限制生成可修改的边集合"""
        modifiable = set()
        for u in range(self.n):
            for v in range(u + 1, self.n):
                try:
                    if nx.shortest_path_length(self.original_graph, u, v) <= b_hop:
                        modifiable.add((u, v))
                except nx.NetworkXNoPath:
                    continue
        return sorted(list(modifiable))

    def encode(self, matrix: np.ndarray) -> List[int]:
        """将邻接矩阵编码为可修改边上的 0-1 向量"""
        encoded = []
        for u, v in self.edge_list:
            encoded.append(1 if matrix[u][v] == 1 else 0)
        return encoded

    def decode(self, bitvector: List[int]) -> np.ndarray:
        """将编码解码为邻接矩阵"""
        matrix = np.zeros((self.n, self.n), dtype=int)
        for i, bit in enumerate(bitvector):
            if bit == 1:
                u, v = self.edge_list[i]
                matrix[u][v] = matrix[v][u] = 1
        return matrix

    def get_edge_list(self) -> List[Tuple[int, int]]:
        return self.edge_list
