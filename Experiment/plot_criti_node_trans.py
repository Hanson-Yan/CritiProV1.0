import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class DrawTopology:
    def __init__(self, matrix, critical_nodes):
        """
        初始化网络拓扑类。
        :param matrix: 邻接矩阵（Numpy 数组或列表的列表）
        :param critical_nodes: 关键节点列表（已格式化为 "sX" 形式）
        """
        self.matrix = matrix
        self.critical_nodes = critical_nodes  # 直接使用 sX 格式的关键节点
        self.G = nx.Graph()
        self._build_graph()
    
    @staticmethod
    def load_matrix_from_file(file_path):
        """从文本文件加载邻接矩阵。"""
        with open(file_path, 'r') as f:
            matrix = [list(map(int, line.split())) for line in f]
        return np.array(matrix)
    
    def _build_graph(self):
        """根据邻接矩阵构建网络拓扑图。"""
        num_nodes = len(self.matrix)
        nodes = [f"s{i}" for i in range(num_nodes)]  # 修改节点名称格式为 s0, s1, s2, ...
        
        # 添加节点（默认所有节点为交换机，圆形）
        for node in nodes:
            self.G.add_node(node, color='lightgreen', shape='o')
        
        # 添加边
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self.matrix[i][j] > 0:
                    self.G.add_edge(f"s{i}", f"s{j}")
    
    # def draw(self, pos, ax, title=""):
    #     """在指定的子图上绘制网络拓扑图，并突出显示关键节点。"""
    #     # 绘制普通节点
    #     node_colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
    #     node_size = 200
    #     font_size = 8
    #     nx.draw_networkx_nodes(self.G, pos, nodelist=self.G.nodes(), node_color=node_colors, node_shape='o', node_size=node_size, ax=ax)
        
    #     # 突出显示关键节点
    #     if self.critical_nodes is not None:
    #         nx.draw_networkx_nodes(self.G, pos, nodelist=self.critical_nodes, node_color='red', 
    #                                node_shape='o', node_size=node_size + 50, edgecolors='black', linewidths=2, ax=ax)
        
    #     # 绘制边
    #     nx.draw_networkx_edges(self.G, pos, ax=ax)
        
    #     # 添加标签
    #     nx.draw_networkx_labels(self.G, pos, font_size=font_size, ax=ax)
        
    #     # 设置子图标题
    #     ax.set_title(title, fontsize=10)
    #     ax.axis('off')  # 关闭坐标轴
    def draw(self, pos, ax, title=""):
        """在指定的子图上绘制网络拓扑图，并突出显示关键节点。"""
        # 使用 spring_layout 布局算法，调整参数以减少节点重叠
        pos = nx.spring_layout(self.G, k=0.6, scale=3)  # k 控制节点间距，scale 控制布局范围
        # pos = nx.spring_layout(self.G, k=0.5, iterations=200, scale=2.0)
        # 绘制普通节点
        node_colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
        node_size = 200
        font_size = 8
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.G.nodes(), node_color=node_colors, node_shape='o', node_size=node_size, ax=ax)
        
        # 突出显示关键节点
        if self.critical_nodes is not None:
            nx.draw_networkx_nodes(self.G, pos, nodelist=self.critical_nodes, node_color='red', 
                                node_shape='o', node_size=node_size + 50, edgecolors='black', linewidths=2, ax=ax)
        
        # 绘制边
        nx.draw_networkx_edges(self.G, pos, ax=ax)
        
        # 添加标签
        nx.draw_networkx_labels(self.G, pos, font_size=font_size, ax=ax)
        
        # 设置子图标题
        ax.set_title(title, fontsize=20)
        ax.axis('off')  # 关闭坐标轴
def draw_multiple_topologies(topologies, titles, save_path=None):
    """
    绘制多个网络拓扑图，按列排列。
    :param topologies: 包含多个 DrawTopology 实例的列表
    :param titles: 每个拓扑图的标题列表
    :param save_path: 保存图像的路径（可选）
    """
    num_topologies = len(topologies)
    num_cols = 4  # 每列 4 张图
    num_rows = (num_topologies + num_cols - 1) // num_cols  # 计算需要多少行
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))  # 创建子图布局

    # 如果只有一行，axs 不是二维数组，需要转换为二维数组
    if num_rows == 1:
        axs = [axs]

    for i, topology in enumerate(topologies):
        col = i // num_rows  # 按列增长
        row = i % num_rows
        ax = axs[row][col]

        pos = nx.spring_layout(topology.G, seed=42)  # 使用相同的布局
        # print(titles[i])
        topology.draw(pos, ax, title=titles[i])

    # 隐藏多余的子图
    for i in range(num_topologies, num_rows * num_cols):
        col = i // num_rows
        row = i % num_rows
        axs[row][col].axis('off')

    plt.tight_layout()  # 自动调整子图间距
    if save_path:
        plt.savefig(save_path, format='png', dpi=600)
        print(f"figure save at {save_path}")
    plt.show()
    plt.close('all')

# 示例调用
if __name__ == "__main__":

    name=input("please input model name(critipro,proto,antitomo):")
    topo_num_arr=["1","2","4","3"]
    topo_arr=[]
    criti_node_arr=[]
    for num in topo_num_arr:
        topo_num=f"topo_{num}"
        original_topo=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{topo_num}.txt"
        original_criti_node=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_original.txt"
        original_topo_matrix=np.loadtxt(original_topo)
        original_criti_node_vector=np.loadtxt(original_criti_node)
        original_criti_node_vector_s=[f's{int(num)}' for num in original_criti_node_vector]
        topo_arr.append(original_topo_matrix)
        criti_node_arr.append(original_criti_node_vector_s)

        _topo=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{name}_{topo_num}_confuse_topo.txt"
        _criti_node=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_{name}.txt"
        _topo_matrix=np.loadtxt(_topo)
        _criti_node_vector=np.loadtxt(_criti_node)
        _criti_node_vector_s=[f's{int(num)}' for num in _criti_node_vector]
        topo_arr.append(_topo_matrix)
        criti_node_arr.append(_criti_node_vector_s)

    draw_topology_arr=[]
    for topo_matrix,criti_node_vector in zip(topo_arr,criti_node_arr):
        draw_topology=DrawTopology(topo_matrix,criti_node_vector)
        draw_topology_arr.append(draw_topology)

        # 遍历 num 数组，填充 titles 数组
    t_name = ""
    if name == "critipro":
        t_name = "CritiPro"
    elif name == "antitomo":
        t_name = "AntiTomo"
    elif name == "proto":
        t_name = "ProTO"
    topo_num_arr=["1","2","3","4"]
    titles = [None] * 8
    for i in range(len(topo_num_arr)):
        # 偶数位置
        titles[i * 2] = f"Original Topology {topo_num_arr[i]}"
        # 奇数位置
        # print(f"name {name}")
        # print(f"{name} Topology {topo_num_arr[i]}")
        titles[i * 2 + 1] = f"{t_name} Topology {topo_num_arr[i]}"
    draw_multiple_topologies(draw_topology_arr, titles, save_path=f"/home/retr0/Project/TopologyObfu/Experiment/{name} multiple topologies.png")


