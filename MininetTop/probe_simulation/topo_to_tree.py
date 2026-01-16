from collections import deque
import numpy as np
from draw_topo import DrawTopology 


class ShortestPathTree:
    def __init__(self, adj_matrix, topo_num):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.parents = self._bfs()
        self.tree_adj_matrix = self._build_tree_adj_matrix()
        self.topo_num=topo_num
    
    def _bfs(self):
        """执行BFS，找到各节点的父节点"""
        n = self.n
        visited = [False] * n
        parents = [None] * n
        queue = deque()
        
        # 根节点（第0行）初始化
        queue.append(0)
        visited[0] = True
        parents[0] = -1  # 根节点无父节点
        
        while queue:
            node = queue.popleft()
            # 遍历邻接节点
            for neighbor in range(n):
                if self.adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                    visited[neighbor] = True
                    parents[neighbor] = node
                    queue.append(neighbor)
        return parents
    
    def _build_tree_adj_matrix(self):
        """根据父节点列表生成树的邻接矩阵"""
        n = self.n
        tree_adj = [[0] * n for _ in range(n)]
        for i in range(1, n):  # 根节点无需处理
            p = self.parents[i]
            if p is not None and p != -1:
                tree_adj[i][p] = 1
                tree_adj[p][i] = 1
        return tree_adj
    
    def get_tree_adj_matrix(self):
        """返回生成的最短路径树的邻接矩阵"""
        return self.tree_adj_matrix

    def save_tree_adj_matrix(self):
        if(self.topo_num!=None):
            save_path="/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"+self.topo_num+".txt"
            np.savetxt(save_path,self.tree_adj_matrix,fmt="%d")

    def save_tree_info(self):
        """保存树的结构信息，包括节点总数、叶子节点数和叶子节点编号"""
        n = self.n
        degree = [sum(row) for row in self.tree_adj_matrix]
        leaf_nodes = [i for i, deg in enumerate(degree) if deg == 1 and i != 0]  # 0不是叶子（根）
        leaf_nodes.insert(0,0)
        host_num=len(leaf_nodes)
        switch_num=n
        connect_switch_order=leaf_nodes
        # 将输入的信息存储到字典中
        user_data = {
            "host_num": host_num,
            "switch_num": switch_num,
            "connect_switch_order": connect_switch_order
        }

        # 将字典写入到文件中
        file_name = f"/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/{self.topo_num}_info.txt"
        with open(file_name, "w") as file:
            for key, value in user_data.items():
                if isinstance(value, list):
                    # 如果值是列表，将其转换为字符串形式
                    file.write(f"{key}: {', '.join(map(str, value))}\n")
                else:
                    file.write(f"{key}: {value}\n")


# 示例用法
if __name__ == "__main__":
    # 示例邻接矩阵（连通图）
    topo_num = input("please input topo_num:")
    adj_matrix = np.loadtxt(f"/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/{topo_num}.txt")
    tree_topo_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"+topo_num+".png"
    DrawTopology(matrix=adj_matrix,critical_nodes=None).draw(show=True)
    spt = ShortestPathTree(adj_matrix,topo_num)
    tree_adj = spt.get_tree_adj_matrix()
    spt.save_tree_adj_matrix()
    spt.save_tree_info()
    DrawTopology(matrix=tree_adj,critical_nodes=None).draw(show=True,save_path=tree_topo_path)
    print("生成的最短路径树的邻接矩阵：")
    for row in tree_adj:
        print(row)