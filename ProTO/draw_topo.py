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
    
    def draw(self, save_path=None):
        """绘制网络拓扑图，并突出显示关键节点。"""
        pos = nx.spring_layout(self.G, seed=42)
        
        # 绘制普通节点
        node_colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.G.nodes(), node_color=node_colors, node_shape='o')
        
        # 突出显示关键节点
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.critical_nodes, node_color='red', 
                               node_shape='o', node_size=500, edgecolors='black', linewidths=2)
        
        # 绘制边
        nx.draw_networkx_edges(self.G, pos)
        
        # 添加标签
        nx.draw_networkx_labels(self.G, pos)
        
        # 设置图形样式
        plt.style.use('default')
        plt.box(False)
        plt.title("Network Topology with Critical Nodes")
        plt.axis('off')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, format='png', dpi=600)
        
        plt.show()

# # 示例使用
# matrix = DrawTopology.load_matrix_from_file("/home/retr0/Project/TopologyObfu/CritiPro/output_file/topo_matrix_confuse.txt")
# critical_nodes = ['s0', 's1']

# topology = DrawTopology(matrix, critical_nodes)
# topology.draw("topology.png")
