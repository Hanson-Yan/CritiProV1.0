import numpy as np

class ObfuscatedTopologyGenerator:
    def __init__(self, original_adj_matrix):
        """
        :param original_adj_matrix: 原始拓扑的邻接矩阵
        """
        self.original_adj_matrix = original_adj_matrix
        self.n = original_adj_matrix.shape[0]  # 获取节点数
    
    def generate_obfuscated_adj_matrix(self):
        """
        生成一个随机的混淆拓扑邻接矩阵，保证其是连通的。
        :return: 混淆拓扑的邻接矩阵
        """
        while True:
            adj_matrix = np.random.randint(0, 2, (self.n, self.n))  # 生成随机邻接矩阵
            np.fill_diagonal(adj_matrix, 0)  # 对角线设为 0，避免自环
            adj_matrix = np.triu(adj_matrix)  # 取上三角，保证对称
            adj_matrix += adj_matrix.T  # 生成无向图
            
            if self.is_connected(adj_matrix):  # 只有连通时才返回
                return adj_matrix
    
    def is_connected(self, adj_matrix):
        """
        使用深度优先搜索 (DFS) 检查图是否连通。
        :param adj_matrix: 邻接矩阵
        :return: True 表示连通，False 表示不连通
        """
        visited = set()
        self.dfs(0, adj_matrix, visited)
        return len(visited) == self.n
    
    def dfs(self, node, adj_matrix, visited):
        """
        深度优先遍历
        """
        visited.add(node)
        for neighbor, connected in enumerate(adj_matrix[node]):
            if connected and neighbor not in visited:
                self.dfs(neighbor, adj_matrix, visited)
