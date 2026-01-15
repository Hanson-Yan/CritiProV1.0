import numpy as np
from collections import defaultdict, deque
import itertools

class AdjacencyToRouting:
    """
    将邻接矩阵转换为基于节点的路由矩阵
    
    物理模型：
    - 延迟主要来自交换机（节点）的处理和排队
    - 链路的传输延迟可忽略
    - 路由矩阵 M[i,j] = 1 表示 receiver pair i 的共享路径经过节点 j
    """
    
    def __init__(self, adjacency_matrix, receiver_pairs=None, root=0):
        """
        参数：
            adjacency_matrix: 原始拓扑的邻接矩阵
            receiver_pairs: 接收器对列表，如 [(2,4), (2,6), ...]
            root: 根节点编号（默认为0）
        """
        self.adj_matrix = np.array(adjacency_matrix)
        self.root = root
        self.tree = self._build_bfs_tree()
        self.num_nodes = len(adjacency_matrix)
        
        # 如果没有提供receiver_pairs，使用叶子节点的所有配对
        if receiver_pairs is None:
            leaves = self._find_leaves()
            self.receiver_pairs = list(itertools.combinations(leaves, 2))
        else:
            self.receiver_pairs = receiver_pairs
    
    def _build_bfs_tree(self):
        """使用BFS构建树结构"""
        n = len(self.adj_matrix)
        tree = defaultdict(list)
        visited = [False] * n
        queue = deque([self.root])
        visited[self.root] = True
        
        while queue:
            u = queue.popleft()
            for v in range(n):
                if self.adj_matrix[u][v] == 1 and not visited[v]:
                    tree[u].append(v)
                    visited[v] = True
                    queue.append(v)
        
        return dict(tree)
    
    def _find_leaves(self):
        """找到所有叶子节点（边缘节点）"""
        all_nodes = set(range(len(self.adj_matrix)))
        internal_nodes = set(self.tree.keys())
        leaves = sorted(all_nodes - internal_nodes - {self.root})
        
        # 特殊情况：如果根节点没有子节点，也算叶子
        if self.root not in internal_nodes and self.root not in leaves:
            leaves.append(self.root)
        
        return leaves
    
    def _get_node_path(self, node):
        """
        获取从节点到根节点的所有节点路径
        
        参数：
            node: 目标节点
            
        返回：
            节点列表，从目标节点到根节点（包括两端）
        """
        path = []
        current = node
        
        # 向上遍历到根节点
        while current != self.root:
            path.append(current)
            
            # 找父节点
            parent = None
            for p, children in self.tree.items():
                if current in children:
                    parent = p
                    break
            
            if parent is None:
                # 如果找不到父节点，说明节点不在树中
                raise ValueError(f"节点 {current} 不在树结构中")
            
            current = parent
        
        # 添加根节点
        path.append(self.root)
        return path
    
    def build_routing_matrix(self):
        """
        构建基于节点的路由矩阵
        
        返回：
            M: numpy数组，形状为 (num_pairs, num_nodes)
            M[i,j] = 1 表示 receiver pair i 的共享路径经过节点 j
        """
        num_pairs = len(self.receiver_pairs)
        num_nodes = self.num_nodes
        routing_matrix = np.zeros((num_pairs, num_nodes), dtype=int)
        
        for idx, (r1, r2) in enumerate(self.receiver_pairs):
            # 获取两个接收器到根节点的节点路径
            try:
                nodes1 = set(self._get_node_path(r1))
                nodes2 = set(self._get_node_path(r2))
            except ValueError as e:
                print(f"警告: {e}")
                continue
            
            # 找出共享节点
            shared_nodes = nodes1 & nodes2
            
            # 在路由矩阵中标记共享节点
            for node in shared_nodes:
                routing_matrix[idx, node] = 1
        
        return routing_matrix
    
    def save_routing_matrix(self, matrix, output_path):
        """保存路由矩阵到文件"""
        np.savetxt(output_path, matrix, fmt='%d')
        print(f"路由矩阵已保存到: {output_path}")
    
    def get_info(self):
        """返回转换信息"""
        return {
            'num_nodes': self.num_nodes,
            'num_pairs': len(self.receiver_pairs),
            'receiver_pairs': self.receiver_pairs,
            'tree_structure': self.tree,
            'leaves': self._find_leaves()
        }
    
    def verify_against_delays(self, delay_vector, unit_delay=12.0):
        """
        验证路由矩阵的正确性
        
        参数：
            delay_vector: 实际测量的延迟向量（上三角）
            unit_delay: 单个节点的处理延迟
            
        返回：
            验证报告字典
        """
        M = self.build_routing_matrix()
        
        # 构建节点延迟向量（假设所有节点延迟相同）
        d_node = np.full(self.num_nodes, unit_delay)
        
        # 计算预期延迟
        predicted_delays = M @ d_node
        
        # 转换上三角延迟矩阵为向量
        n = len(self._find_leaves())
        delay_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                delay_matrix[i][j] = delay_vector[idx] if idx < len(delay_vector) else 0
                idx += 1
        
        actual_delays = []
        for i in range(n):
            for j in range(i+1, n):
                actual_delays.append(delay_matrix[i][j])
        actual_delays = np.array(actual_delays[:len(self.receiver_pairs)])
        
        # 计算误差
        errors = actual_delays - predicted_delays
        
        return {
            'predicted': predicted_delays,
            'actual': actual_delays,
            'errors': errors,
            'mean_error': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2))
        }


def get_user_input(info_path):
    """
    从拓扑信息文件中读取配置
    
    参数：
        info_path: 拓扑信息文件路径
        
    返回：
        host_num, switch_num, connect_switch_order
    """
    with open(info_path, 'r') as f:
        lines = f.readlines()
    
    # 解析文件内容（根据您的文件格式调整）
    host_num = int(lines[0].strip().split(':')[1])
    switch_num = int(lines[1].strip().split(':')[1])
    connect_switch_order = [int(x) for x in lines[2].strip().split(':')[1].split(',')]
    
    return host_num, switch_num, connect_switch_order


def run_ad_to_rout(adj_matrix_path, output_routing_path, root_node, receiver_nodes):
    """
    邻接矩阵转路由矩阵的便捷接口
    
    参数：
        adj_matrix_path: 邻接矩阵文件路径
        output_routing_path: 输出路由矩阵路径
        root_node: 根节点编号
        receiver_nodes: 接收器节点列表
    """
    # 读取邻接矩阵
    adj_matrix = np.loadtxt(adj_matrix_path, dtype=int)
    
    # 生成接收器对
    import itertools
    receiver_pairs = list(itertools.combinations(receiver_nodes, 2))
    
    # 创建转换器
    converter = AdjacencyToRouting(adj_matrix, receiver_pairs, root=root_node)
    
    # 构建路由矩阵
    M = converter.build_routing_matrix()
    
    # 保存
    converter.save_routing_matrix(M, output_routing_path)
    
    # 打印信息
    info = converter.get_info()
    print(f"节点数: {info['num_nodes']}")
    print(f"接收器对数: {info['num_pairs']}")
    print(f"路由矩阵形状: {M.shape}")


def main():
    """测试示例"""
    # topo_1 的邻接矩阵
    adj_matrix = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0]
    ]
    
    # 边缘节点配对
    receivers = [2, 4, 6, 7, 8]
    receiver_pairs = list(itertools.combinations(receivers, 2))
    
    # 创建转换器
    converter = AdjacencyToRouting(adj_matrix, receiver_pairs, root=0)
    
    # 获取信息
    info = converter.get_info()
    print("=" * 60)
    print("拓扑信息:")
    print(f"节点数: {info['num_nodes']}")
    print(f"接收器对数: {info['num_pairs']}")
    print(f"叶子节点: {info['leaves']}")
    print(f"树结构: {info['tree_structure']}")
    
    # 构建路由矩阵
    M = converter.build_routing_matrix()
    print("\n" + "=" * 60)
    print("路由矩阵 M (基于节点):")
    print(f"维度: {M.shape} (pairs × nodes)")
    print("\n列标签: ", end="")
    for i in range(M.shape[1]):
        print(f"node{i:2d} ", end="")
    print()
    
    for i, (r1, r2) in enumerate(receiver_pairs):
        print(f"pair{i} ({r1},{r2}): ", end="")
        for j in range(M.shape[1]):
            print(f"  {M[i,j]}    ", end="")
        shared = np.where(M[i] == 1)[0]
        print(f" → 共享节点: {list(shared)}")
    
    # 验证
    print("\n" + "=" * 60)
    print("验证路由矩阵:")
    
    # 您提供的延迟向量
    delay_vector = [
        12.046080, 12.798984, 12.604495, 12.918168,
        27.000000, 24.619000, 26.386000,
        36.945000, 38.879000,
        51.000000
    ]
    
    unit_delay = 12.0  # μs
    
    report = converter.verify_against_delays(delay_vector, unit_delay)
    
    print(f"\n{'Pair':<10} {'预期(μs)':<12} {'实际(μs)':<12} {'误差(μs)':<12}")
    print("-" * 50)
    for i, (r1, r2) in enumerate(receiver_pairs):
        print(f"({r1},{r2}){' '*5} {report['predicted'][i]:>10.2f}  "
              f"{report['actual'][i]:>10.2f}  {report['errors'][i]:>10.2f}")
    
    print(f"\n统计:")
    print(f"平均绝对误差: {report['mean_error']:.2f} μs")
    print(f"最大误差: {report['max_error']:.2f} μs")
    print(f"RMSE: {report['rmse']:.2f} μs")
    
    # 保存路由矩阵
    converter.save_routing_matrix(M, "routing_matrix_topo1.txt")


if __name__ == "__main__":
    main()
