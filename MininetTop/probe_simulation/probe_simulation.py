import numpy as np
from itertools import combinations
from collections import deque
from draw_topo import DrawTopology 
import os

class ProbeSimulation:
    def __init__(self, adjacency_matrix, bw, probe_num,topo_num):
        self.adj_matrix = adjacency_matrix
        self.bw = bw
        self.probe_num = probe_num
        self.MTU = 1500  # 常量MTU
        self.unit_delay= (self.MTU * 8) / self.bw
        self.parent, self.children = self._build_tree()
        self.leaves = sorted(self._find_leaves())  # 排序叶子节点
        # self.leaves = self._find_leaves()  # 排序叶子节点
        self.delay_matrix,self.base_matrix = self._calculate_delays()
        self.topo_num= topo_num

    def _build_tree(self):
        """构建树结构并返回父节点和子节点列表"""
        n = len(self.adj_matrix)
        parent = [-1] * n
        children = [[] for _ in range(n)]
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if self.adj_matrix[u][v] == 1 and not visited[v]:
                    parent[v] = u
                    children[u].append(v)
                    visited[v] = True
                    queue.append(v)
        return parent, children

    def _find_leaves(self):
        """找到所有叶子节点（没有子节点的节点）"""
        return [i for i in range(len(self.children)) if not self.children[i]]

    def _get_ancestry(self, node):
        """获取从节点到根节点的路径"""
        path = []
        while node != -1:
            path.append(node)
            node = self.parent[node]
        return path

    def _calculate_delays(self):
        """计算延迟矩阵（上三角）"""
        n = len(self.leaves)
        # 初始化全零矩阵
        matrix_simu = [[0.0 for _ in range(n)] for _ in range(n)]
        matrix_base = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # 预计算所有叶子节点的路径
        paths = {leaf: set(self._get_ancestry(leaf)) for leaf in self.leaves}
        
        # 遍历所有叶子节点对
        for i in range(n):
            for j in range(i+1, n):
                u = self.leaves[i]
                v = self.leaves[j]
                
                # 找出共享路径
                common_nodes = paths[u] & paths[v]
                node_count = len(common_nodes)
                
                # 计算基础延迟
                base_delay = self.unit_delay * node_count
                
                matrix_base[i][j] = base_delay
                # 添加高斯噪声（噪声标准差与探测次数相关）
               
                # 设置噪声随 probe_num 的衰减参数

                # 参数设置
                target_probe = 10000
                initial_std_ratio = 0.12  # 初期波动放大（更容易达到 +3）
                final_std_ratio = 0.002
                gamma = 2.5
                max_error_abs = 3.0
                min_error_abs = 0.3  # 保底扰动
                epsilon = 1e-6  # 防止误差为 0

                # 收敛进度控制（前陡后缓）
                progress = (np.log10(self.probe_num + 10) / np.log10(target_probe + 10)) ** gamma

                # 误差标准差，插值后控制上下限
                std_ratio = initial_std_ratio * (1 - progress) + final_std_ratio * progress
                noise_std = np.clip(base_delay * std_ratio, min_error_abs, max_error_abs)

                # 偏置控制：仍保留一定测量方向性
                bias_amplitude = base_delay * 0.015 * (1 - progress)
                bias = np.random.uniform(0, bias_amplitude)  # 始终为正偏置（模拟总是偏大）

                # 生成扰动值：截断高斯分布为正数，再加偏置
                raw_noise = np.random.normal(0, noise_std)
                positive_noise = np.clip(abs(raw_noise) + bias, 0, max_error_abs)

                # 确保最终扰动不超过链路能力
                max_allowed_error = self.MTU * 8 / self.bw / 2
                final_noise = np.clip(positive_noise, epsilon, max_allowed_error)

                # 计算最终延迟（确保 > base_delay）
                final_delay = base_delay + final_noise
                matrix_simu[i][j] = final_delay
                
        return matrix_simu,matrix_base

    def get_delay_matrix(self):
        """获取最终的延迟矩阵"""
        return self.delay_matrix
    def save_result(self):
        delay_dir="/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/"
        delay_dir_probenum=delay_dir+str(self.probe_num)
        if not os.path.exists(delay_dir_probenum):
            # 如果目录不存在，则创建目录
            os.makedirs(delay_dir_probenum)
            print(f"目录 '{delay_dir_probenum}' 已创建。")
        simu_result_path=delay_dir_probenum+"/"+self.topo_num+"_simu_delay.txt"
        base_result_path=delay_dir+self.topo_num+"_base_delay.txt"
        np.savetxt(simu_result_path,self.delay_matrix,fmt="%.6f")
        np.savetxt(base_result_path,self.base_matrix,fmt="%.6f")


# 示例用法
def probe_simu(topo_num,probe_num=100,bw = 1000 ):
    # 示例邻接矩阵（0是根节点，0-1-2，0-1-3，0-1-4结构）
    adj_matrix = np.loadtxt("/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"+topo_num+".txt")
   
    DrawTopology(matrix=adj_matrix,critical_nodes=None).draw()
    simulator = ProbeSimulation(adj_matrix, bw, probe_num,topo_num)
    simulator.save_result()
    print(simulator._find_leaves())
    print("延迟矩阵:")
    print(simulator.get_delay_matrix())