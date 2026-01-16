# import numpy as np
# from collections import deque

# class AdjacencyToRouting:
#     def __init__(self, file_path=None, adj_matrix=None):
#         if file_path:
#             self.adj_matrix = self._load_adj_matrix(file_path)
#         elif adj_matrix is not None:
#             self.adj_matrix = np.array(adj_matrix)
#         else:
#             raise ValueError("必须提供邻接矩阵或文件路径")
            
#         self.num_nodes = len(self.adj_matrix)
#         self.root = 0
#         self.tree = self._build_bfs_tree()
#         self.link_map = self._assign_link_ids()
#         self.receiver_pairs = self._get_ordered_receiver_pairs()
        
#     def _load_adj_matrix(self, file_path):
#         with open(file_path, 'r') as f:
#             return np.array([[int(x) for x in line.strip().split()] for line in f])
        

    
#     def _build_bfs_tree(self):
#         """BFS构建树结构，严格按层级和列号顺序"""
#         tree = {}
#         visited = set([self.root])
#         queue = deque([self.root])
#         tree[self.root] = []
        
#         while queue:
#             parent = queue.popleft()
#             # 获取子节点并严格按列号排序
#             children = np.where(self.adj_matrix[parent] == 1)[0].tolist()
#             children = sorted([c for c in children if c not in visited and c != parent])
#             tree[parent] = children
#             for child in children:
#                 visited.add(child)
#                 queue.append(child)
#                 tree[child] = []  # 初始化子节点
#         return tree
    
#     def _assign_link_ids(self):
#         """严格BFS顺序分配链路ID"""
#         link_id = 0
#         link_map = {}
#         queue = deque([self.root])
        
#         while queue:
#             parent = queue.popleft()
#             for child in sorted(self.tree[parent]):  # 按列号排序
#                 link_map[(parent, child)] = link_id
#                 link_id += 1
#                 queue.append(child)
#         return link_map
    
#     def _get_ordered_receiver_pairs(self):
#         """按DFS顺序生成接收器对 (7,8)->(7,6)->(7,4)..."""
#         # 深度优先遍历获取叶子节点顺序
#         leaves = []
        
#         def dfs(node):
#             if not self.tree[node]:  # 叶子节点
#                 leaves.append(node)
#                 return
#             # 按列号顺序遍历子节点 (确保从左到右)
#             for child in sorted(self.tree[node]):
#                 dfs(child)
        
#         dfs(self.root)  # 从根开始遍历
#         leaves = [n for n in leaves if n != self.root]  # 排除根节点
        
#         # 生成严格有序的接收器对
#         ordered_pairs = []
#         for i in range(len(leaves)):
#             for j in range(i+1, len(leaves)):
#                 ordered_pairs.append( (leaves[i], leaves[j]) )
        
#         return ordered_pairs
    
#     # def _get_bfs_order(self, node):
#     #     """获取节点在BFS遍历中的出现顺序"""
#     #     visited = set()
#     #     queue = deque([self.root])
#     #     order = 0
#     #     while queue:
#     #         current = queue.popleft()
#     #         if current == node:
#     #             return order
#     #         visited.add(current)
#     #         for child in sorted(self.tree[current]):
#     #             if child not in visited:
#     #                 queue.append(child)
#     #         order += 1
#     #     return float('inf')
    
#     def _get_full_path(self, node):
#         """获取从节点到根的完整链路路径"""
#         path = []
#         current = node
#         while current != self.root:
#             # 查找直接父节点
#             parents = [p for p in self.tree if current in self.tree[p]]
#             parent = parents[0]
#             path.insert(0, (parent, current))
#             current = parent
#         return path
    
#     def build_routing_matrix(self):
#         num_pairs = len(self.receiver_pairs)
#         num_links = len(self.link_map)
#         routing_matrix = np.zeros((num_pairs, num_links), dtype=int)
        
#         for idx, (r1, r2) in enumerate(self.receiver_pairs):
#             path1 = self._get_full_path(r1)
#             path2 = self._get_full_path(r2)
            
#             # 找出所有共享链路
#             shared_links = set(path1) & set(path2)
            
#             # 标记路由矩阵中的对应位置
#             for link in shared_links:
#                 link_id = self.link_map[link]
#                 routing_matrix[idx, link_id] = 1
                
#         return routing_matrix
    
#     def save_routing_matrix(self, file_path):
#         np.savetxt(file_path, self.build_routing_matrix(), fmt='%d')
#         print("路由矩阵已写入")


# # 验证测试（使用您的9节点示例）
# if __name__ == "__main__":
#     # example_adj = [
#     #     [0,1,1,0,0,0,0,0,0],
#     #     [1,0,0,1,1,0,0,0,0],
#     #     [1,0,0,0,0,0,0,0,0],
#     #     [0,1,0,0,0,1,1,0,0],
#     #     [0,1,0,0,0,0,0,0,0],
#     #     [0,0,0,1,0,0,0,1,1],
#     #     [0,0,0,1,0,0,0,0,0],
#     #     [0,0,0,0,0,1,0,0,0],
#     #     [0,0,0,0,0,1,0,0,0]
#     # ]
#     adj_matrix_path="/home/retr0/Project/TopologyObfu/CritiPro/input_file/topo_matrix_original.txt"
#     routing_matrix_path="/home/retr0/Project/TopologyObfu/CritiPro/output_file/routing_matrix.txt"
    
#     converter = AdjacencyToRouting(file_path=adj_matrix_path)
#     print("链路映射:", converter.link_map)
#     print("recevier pairs:",converter.receiver_pairs)
#     converter.save_routing_matrix(routing_matrix_path)
#     print("路由矩阵:\n", converter.build_routing_matrix())


import numpy as np
from collections import deque

class AdjacencyToRouting:
    def __init__(self, file_path=None, adj_matrix=None, root=0, receivers=None):
        if file_path:
            self.adj_matrix = self._load_adj_matrix(file_path)
        elif adj_matrix is not None:
            self.adj_matrix = np.array(adj_matrix)
        else:
            raise ValueError("必须提供邻接矩阵或文件路径")
            
        self.num_nodes = len(self.adj_matrix)
        self.root = root  # 使用用户指定的根节点
        self.tree = self._build_bfs_tree()
        self.link_map = self._assign_link_ids()
        self.receivers = receivers if receivers else self._get_default_receivers()
        self.receiver_pairs = self._get_ordered_receiver_pairs()
        
    def _load_adj_matrix(self, file_path):
        with open(file_path, 'r') as f:
            return np.array([[int(x) for x in line.strip().split()] for line in f])
    
    def _build_bfs_tree(self):
        """根据用户指定的根节点构建树结构"""
        tree = {}
        visited = set([self.root])
        queue = deque([self.root])
        tree[self.root] = []
        
        while queue:
            parent = queue.popleft()
            # 获取子节点并严格按列号排序
            children = np.where(self.adj_matrix[parent] == 1)[0].tolist()
            children = sorted([c for c in children if c not in visited and c != parent])
            tree[parent] = children
            for child in children:
                visited.add(child)
                queue.append(child)
                tree[child] = []  # 初始化子节点
        return tree
    
    def _assign_link_ids(self):
        """严格BFS顺序分配链路ID"""
        link_id = 0
        link_map = {}
        queue = deque([self.root])
        
        while queue:
            parent = queue.popleft()
            for child in sorted(self.tree[parent]):  # 按列号排序
                link_map[(parent, child)] = link_id
                link_id += 1
                queue.append(child)
        return link_map
    
    def _get_default_receivers(self):
        """获取默认叶子节点（当用户未指定时）"""
        return [node for node in self.tree if not self.tree[node] and node != self.root]
    
    def _get_ordered_receiver_pairs(self):
        """根据用户指定的接收节点顺序生成配对"""
        if not self.receivers:
            return []
        
        # 验证接收节点有效性
        valid_receivers = [n for n in self.receivers if n in range(self.num_nodes)]
        if len(valid_receivers) != len(self.receivers):
            print("警告：部分接收节点无效，已自动过滤")
        
        # 生成严格有序的接收器对
        ordered_pairs = []
        for i in range(len(valid_receivers)):
            for j in range(i+1, len(valid_receivers)):
                ordered_pairs.append( (valid_receivers[i], valid_receivers[j]) )
        return ordered_pairs
    
    def _get_full_path(self, node):
        """获取从节点到根的完整链路路径"""
        path = []
        current = node
        while current != self.root:
            # 查找直接父节点
            parents = [p for p in self.tree if current in self.tree[p]]
            if not parents:
                raise ValueError(f"节点{current}无法连接到根节点{self.root}")
            parent = parents[0]
            path.insert(0, (parent, current))
            current = parent
        return path
    
    def build_routing_matrix(self):
        num_pairs = len(self.receiver_pairs)
        num_links = len(self.link_map)
        routing_matrix = np.zeros((num_pairs, num_links), dtype=int)
        
        for idx, (r1, r2) in enumerate(self.receiver_pairs):
            path1 = self._get_full_path(r1)
            path2 = self._get_full_path(r2)
            
            # 找出所有共享链路
            shared_links = set(path1) & set(path2)
            
            # 标记路由矩阵中的对应位置
            for link in shared_links:
                link_id = self.link_map[link]
                routing_matrix[idx, link_id] = 1
                
        return routing_matrix
    
    def save_routing_matrix(self, file_path):
        np.savetxt(file_path, self.build_routing_matrix(), fmt='%d')
        print(f"路由矩阵已成功保存到 {file_path}")

def get_user_input(file_path):
    # """获取用户输入"""
    
    # root = int(input("请输入根节点编号（0开始的整数）: "))
    # receivers = list(map(int, input("请按顺序输入接收器连接的节点编号（用空格分隔）: ").split()))
    # return root, receivers
    file_name = file_path
    # 初始化变量
    host_num = None
    switch_num = None
    connect_switch_order = None

    try:
        # 打开文件并逐行读取
        with open(file_name, "r") as file:
            for line in file:
                # 去除行首行尾的空白字符
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                # 分割键和值
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "host_num":
                    host_num = int(value)
                elif key == "switch_num":
                    switch_num = int(value)
                elif key == "connect_switch_order":
                    # 将字符串转换为整数数组
                    connect_switch_order = list(map(int, value.split(",")))

        # 检查是否成功读取所有必要的数据
        if host_num is None or switch_num is None or connect_switch_order is None:
            raise ValueError("文件中缺少必要的信息！")

        return host_num, switch_num, connect_switch_order

    except FileNotFoundError:
        print(f"错误：文件 {file_name} 未找到！")
        return None, None, None
    except ValueError as e:
        print(f"错误：读取文件时发生错误 - {e}")
        return None, None, None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None, None, None

def run_ad_to_rout(adj_matrix_path,routing_matrix_path,root,receivers):
    # 文件路径配置
    # adj_matrix_path = "/home/retr0/Project/TopologyObfu/CritiPro/input_file/topo_matrix_original.txt"
    # routing_matrix_path = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/routing_matrix.txt"
    
    try:
        # 初始化转换器
        converter = AdjacencyToRouting(
            file_path=adj_matrix_path,
            root=root,
            receivers=receivers
        )
        
        print("\n链路映射:", converter.link_map)
        print("接收节点对:", converter.receiver_pairs)
        print("生成的路由矩阵:\n", converter.build_routing_matrix())
        
        # 保存结果
        converter.save_routing_matrix(routing_matrix_path)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("请检查：")
        print("1. 根节点是否有效")
        print("2. 接收节点是否可达")
        print("3. 输入文件格式是否正确")

if __name__ == "__main__":
    adj_matrix_path = "/home/retr0/Project/TopologyObfu/CritiPro/input_file/topo_matrix_original.txt"
    routing_matrix_path = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/routing_matrix.txt"
    # 获取用户输入
    root, receivers = get_user_input()
    run_ad_to_rout(adj_matrix_path,routing_matrix_path,root,receivers)