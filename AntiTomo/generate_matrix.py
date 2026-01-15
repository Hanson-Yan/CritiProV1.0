import numpy as np
import pulp
import time
from collections import deque
import itertools
import random
from adjacency_to_routing import AdjacencyToRouting

class AntiTomoDefender:
    def __init__(self, original_adj, original_delays, destination_pair,params,internal_link_delay = 10):
        """
        :param original_adj: 原始拓扑邻接矩阵 (numpy矩阵)
        :param original_delays: 原始链路时延向量 (numpy数组)
        :param D: 目的节点列表
        :param params: 参数字典，包含：
            phi_n, phi_l, v_min, v_max, delta_max,
            lambda_simi, lambda_cost, sigma,
            w_candidates (候选树数量)
        """
        self.original_adj = original_adj
        self.original_delays = original_delays
        self.D = destination_pair
        self.params = params
        self.internal_link_delay = internal_link_delay
        
        # 预处理原始拓扑信息
        self.N = original_adj.shape[0]
        self.L = sum(original_adj[i][j] != 0 for i in range(len(original_adj)) for j in range(i + 1, len(original_adj)))
        self.original_tree = self._build_original_tree()
        
    def _build_original_tree(self):
        """构建原始拓扑的树结构表示"""
        converter = AdjacencyToRouting(adj_matrix=self.original_adj, root=0, receivers=self.D)
        return {
            'adj_matrix': self.original_adj,
            'routing_matrix': converter.build_routing_matrix(),
            'link_map': converter.link_map,
            'receiver_pairs': converter.receiver_pairs
        }
    
    def _pad_matrices(self, matrix1, matrix2):
        """统一矩阵维度并进行零填充"""
        max_size = max(matrix1.shape[0], matrix2.shape[0])
        padded1 = np.zeros((max_size, max_size), dtype=int)
        padded1[:matrix1.shape[0], :matrix1.shape[1]] = matrix1
        padded2 = np.zeros((max_size, max_size), dtype=int)
        padded2[:matrix2.shape[0], :matrix2.shape[1]] = matrix2
        return padded1, padded2

    def _matrix_levenshtein_distance(self, matrix1, matrix2):
        """计算两个矩阵之间的Levenshtein距离"""
        rows1, cols1 = matrix1.shape
        rows2, cols2 = matrix2.shape
        
        # 确保矩阵维度相同
        assert rows1 == rows2 and cols1 == cols2, "矩阵维度必须相同"
        
        # 初始化距离矩阵
        d = np.zeros((rows1 + 1, rows2 + 1), dtype=int)
        
        # 初始化边界条件
        for i in range(rows1 + 1):
            d[i, 0] = i
        for j in range(rows2 + 1):
            d[0, j] = j
            
        # 动态规划填充
        for i in range(1, rows1 + 1):
            for j in range(1, rows2 + 1):
                cost = 0 if np.array_equal(matrix1[i-1], matrix2[j-1]) else 1
                d[i, j] = min(d[i-1, j] + 1,    # 删除行
                              d[i, j-1] + 1,    # 插入行
                              d[i-1, j-1] + cost) # 替换行
        return d[rows1, rows2]

    def _calculate_similarity(self, candidate):
        """计算拓扑相似性（论文公式5-4）"""
        # 获取矩阵并统一维度
        original_padded, candidate_padded = self._pad_matrices(
            self.original_tree['adj_matrix'], 
            candidate['adj_matrix']
        )
        
        # 计算Levenshtein距离
        LD = self._matrix_levenshtein_distance(original_padded, candidate_padded)
        
        # 计算到空树的距离
        empty_matrix = np.zeros_like(original_padded)
        ted_t_z = self._matrix_levenshtein_distance(original_padded, empty_matrix)
        ted_tprime_z = self._matrix_levenshtein_distance(candidate_padded, empty_matrix)
        
        # 处理除零情况
        denominator = ted_t_z + ted_tprime_z
        if denominator == 0:
            return 1.0
        
        return 1 - (LD / denominator)


    def generate_obfuscated_topology(self):
        """生成混淆拓扑主流程"""
        # 步骤1：生成候选森林
        candidate_forest = self._generate_candidate_forest()
        
        # 步骤2：筛选并评估候选树
        best_candidate = None
        min_objective = float('inf')
        delay_vector = []
        candidate_num = 0

        for candidate in candidate_forest:
            # 约束检查
            candidate_num += 1
            print(f"检查候选树{candidate_num}")
            if not self._check_candidate_constraints(candidate):
                continue
            # 求解线性规划
            lp_status, cost, relate_delay = self._solve_lp(candidate)
            if lp_status != pulp.LpStatusOptimal:
                continue
              
            # 计算相似性
            similarity = self._calculate_similarity(candidate)
            
            # 计算目标函数
            objective = self.params['lambda_simi'] * similarity + self.params['lambda_cost'] * cost
             
            if objective < min_objective: 
                min_objective = objective
                best_candidate = candidate
                delay_vector = relate_delay.copy()
        return best_candidate, min_objective, delay_vector
    
    def _generate_candidate_forest(self):
        """候选森林生成算法（算法1）"""
        forest = []
        n_lower = self.N - self.params['phi_n']
        n_upper = self.N + self.params['phi_n']
        print(f"生成候选森林，节点数范围: [{n_lower}, {n_upper}]，目标数量: {self.params['w_candidates']}")
        attempt = 0
        while len(forest) < self.params['w_candidates']:
            # 随机生成节点数
            attempt += 1
            n = random.randint(n_lower, n_upper)
            
            # 生成随机树
            candidate = self._generate_random_tree(n)
            if candidate is None:
                print(f"尝试 {attempt}: 生成失败")
                continue
                
            # 检查目的节点数量
            if len(candidate['D']) != len(self.D):
                print(f"尝试 {attempt}: 目的节点数量不符 {len(candidate['D'])} vs {len(self.D)}")
                continue
            print(f"成功生成候选树 {len(forest)+1}: 节点数={n}, 目的节点={candidate['D']}")
            forest.append(candidate)
        print(f"候选森林生成完成，有效树数量: {len(forest)}")
        return forest
    
    # def _generate_random_tree(self, n_nodes):
    #     """生成随机树结构"""
    #     nodes = list(range(n_nodes))
    #     tree = {i: [] for i in nodes}
    #     edges = []
    #     connected = set([0])
        
    #     while len(edges) < n_nodes - 1:
    #         # 随机选择两个节点
    #         s, d = random.sample(nodes, 2)
    #         if s == d:
    #             continue
                
    #         # 检查路径是否存在
    #         if not self._has_path(tree, s, d):
    #             tree[s].append(d)
    #             edges.append((s, d))
    #             connected.add(d)
                
    #     # 提取目的节点（假设与原始拓扑相同位置）
    #     D_candidate = [i for i in nodes if i in self.D and i < n_nodes]
        
    #     return {
    #         'adj_matrix': self._tree_to_adj_matrix(tree, n_nodes),
    #         'D': D_candidate,
    #         'tree_struct': tree
    #     }
    def _generate_random_tree(self, n_nodes):
        """改进的随机树生成算法"""
        nodes = list(range(n_nodes))
        parent = np.random.choice(nodes)
        tree = {i: [] for i in nodes}
        connected = {parent}
        unconnected = set(nodes) - connected
        
        while unconnected:
            child = random.choice(list(unconnected))
            tree[parent].append(child)
            connected.add(child)
            unconnected.remove(child)
            parent = random.choice(list(connected))  # 随机选择已连接节点作为父节点
        
        # 提取目的节点（修正索引越界问题）
        D_candidate = [i for i in self.D if i < n_nodes]
        
        return {
            'adj_matrix': self._tree_to_adj_matrix(tree, n_nodes),
            'D': D_candidate,
            'tree_struct': tree
        }
    
    def _tree_to_adj_matrix(self, tree, n_nodes):
        """将树结构转换为邻接矩阵"""
        adj = np.zeros((n_nodes, n_nodes), dtype=int)
        for parent in tree:
            for child in tree[parent]:
                adj[parent, child] = 1
                adj[child, parent] = 1
        return adj
    
    def _check_candidate_constraints(self, candidate):
        """检查候选树约束"""

        # 节点数差异
        if abs(len(candidate['tree_struct']) - self.N) >= self.params['phi_n']:
            print("节点数差异不满足")
            return False
            
        # 链路数差异
        edge_count = sum(len(v) for v in candidate['tree_struct'].values())
        if abs(edge_count - self.L) >= self.params['phi_l']:
            print("链路数差异不满足")
            return False
            
        # 度数方差检查
        degrees = [len(children) for children in candidate['tree_struct'].values()]
        var = np.var(degrees)
        if not (self.params['v_min'] <= var <= self.params['v_max']):
            print("度数方差检查不满足")
            return False
            
        return True
    
    def _solve_lp(self, candidate):
        """求解线性规划问题（公式5 12）"""
        # 构建候选树路由矩阵
        converter = AdjacencyToRouting(
            adj_matrix=candidate['adj_matrix'],
            root=0,
            receivers=candidate['D']
        )
        R_prime = converter.build_routing_matrix()
        
        print(f"候选树路由矩阵维度: {R_prime.shape}，链路数: {len(candidate['adj_matrix'])}")
        
        # 检查路由矩阵是否为空
        if R_prime.size == 0:
            print("错误: 路由矩阵为空")
            return (pulp.LpStatusNotSolved, None)

        # 创建LP问题
        prob = pulp.LpProblem("DelayMinimization", pulp.LpMinimize)
        
        # 决策变量：新链路时延
        # x_prime = [pulp.LpVariable(f"x_{i}", lowBound=self.original_delays[i]) for i in range(R_prime.shape[1])]
        x_prime = []
        for i in range(R_prime.shape[1]):
            # if i < len(self.original_delays):
            #     lb = self.original_delays[i]
            # else:
            #     # 新增链路使用原始时延平均值作为下界
            #     lb = np.mean(self.original_delays)
            lb = self.internal_link_delay
            x_prime.append(pulp.LpVariable(f"x_{i}", lowBound=lb))
        
        # 目标函数：总时延
        prob += pulp.lpSum([pulp.lpDot(R_prime[i], x_prime) for i in range(R_prime.shape[0])])
        
        # 约束：时延不超过最大等待时间
        for i in range(R_prime.shape[0]):
            prob += max(self.original_delays[i], self.internal_link_delay)<= pulp.lpDot(R_prime[i], x_prime) <= self.params['delta_max']
            
        # # 约束：相关时延大于原始相关时延
        # for i in range(R_prime.shape[0]):
        #     prob += pulp.lpDot(R_prime[i], x_prime) >= max(self.original_delays[i], 10)  # 修改下界
        # 求解
        start_time = time.time()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        solve_time = time.time() - start_time
        
        # 检查求解时间
        if solve_time > self.params['sigma']:
            print("求解超时")
            return (pulp.LpStatusNotSolved, None, None)
            
        if status != pulp.LpStatusOptimal:
            print("线性规划求解失败")
            return (status, None, None)
        
        
        # 计算代价
        y_prime = []  # 存储每条链路的相关时延
        cost = 0.0
        original_len = len(self.original_delays)
        avg_delay = np.mean(self.original_delays)
        x_prime_arr = []
        base_delay_arr = []
        for i in range(len(x_prime)):
            x_prime_value = pulp.value(x_prime[i])
            
            # 处理未求解情况
            if x_prime_value is None:
                print(f"警告: 变量x_{i}未正确求解，使用下界值{x_prime[i].lowBound}")
                x_prime_value = x_prime[i].lowBound
            x_prime_arr.append(x_prime_value)
            # 选择基准时延
            base_delay = self.original_delays[i] if i < original_len else avg_delay
            base_delay_arr.append(base_delay)
        
        x_vector=np.array(x_prime_arr)
        for i in range(R_prime.shape[0]):  # 对每一行计算相关时延
            #计算相关时延
            router_vector = np.array(R_prime[i])
            # print(f"x_vector.shape{x_vector.shape}")
            # print(f"router_vector.shape{router_vector.shape}")
            y_ij=np.dot(router_vector,x_vector)
            y_prime.append(y_ij)
            # 计算代价函数
            cost += (y_ij / self.original_delays[i]) - 1
        print(f"y_prime{y_prime}")
        return (status, cost, y_prime)
    
    
    # 辅助方法
    def _has_path(self, tree, s, d):
        """检查树中路径存在性"""
        visited = set()
        stack = [s]
        while stack:
            node = stack.pop()
            if node == d:
                return True
            visited.add(node)
            stack.extend([child for child in tree[node] if child not in visited])
        return False

# 使用示例
if __name__ == "__main__":
    # 示例参数配置
    params = {
        'phi_n': 3,
        'phi_l': 5,
        'v_min': 0.1,
        'v_max': 3.0,
        'delta_max': 100,
        'lambda_simi': 0.7,
        'lambda_cost': 0.3,
        'sigma': 5,
        'w_candidates': 100
    }
    
    # 示例输入数据
    original_adj = np.array([
        [0,1,1,0,0,0,0,0,0],
        [1,0,0,1,1,0,0,0,0],
        [1,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,1,1,0,0],
        [0,1,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,1,1],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0,0]
    ])
    original_delays = np.array([2,3,2,4,1,3,2,2,3])
    D = [4,6,7,8]
    
    # 初始化防御系统
    defender = AntiTomoDefender(original_adj, original_delays, D, params)
    
    # 生成混淆拓扑
    best_candidate,objective = defender.generate_obfuscated_topology()
    
    # 输出结果
    if best_candidate:
        print("最优混淆拓扑邻接矩阵:\n", best_candidate['adj_matrix'])
        print("目标函数值:", objective)
    else:
        print("未找到有效混淆拓扑")