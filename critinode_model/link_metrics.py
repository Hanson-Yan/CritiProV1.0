import networkx as nx

class LinkMetricsCalculator:
    """链路指标计算类，用于计算网络中各链路的关键指标"""
    
    def __init__(self, net, link_bandwidths, traffic_data):
        """
        初始化链路指标计算器
        :param graph: NetworkX图实例
        :param link_bandwidths: 字典，包含每条链路的带宽，格式为{(s1, s2): bandwidth}
        :param traffic_data: 链路流量数据，从外部传入，格式为{(s1,s2): traffic_info}
        """
        self.net = net
        self.graph = self.mininet_to_nx()
        self.link_bandwidths = link_bandwidths
        self.traffic_data = traffic_data  # 初始化时传入并存储流量数据
        self.ll = 20  # 固定链路延迟为20
    
    def mininet_to_nx(self):
        """把 Mininet 网络转换成 NetworkX 图"""
        G = nx.Graph()
        for link in self.net.links:
            node1 = link.intf1.node.name
            node2 = link.intf2.node.name
            # 可选：把带宽等属性加进去
            bw1 = link.intf1.params.get('bw', None)
            bw2 = link.intf2.params.get('bw', None)
            bandwidth = bw1 or bw2 or 100  # 默认 100 Mbps
            G.add_edge(node1, node2, bandwidth=bandwidth)
        return G
         
    def calculate_ebc(self):
        """计算所有链路的边介数中心性，返回{(s1,s2): ebc_value}格式"""
        return nx.edge_betweenness_centrality(self.graph)
    
    def calculate_cbr(self):
        """
        计算已消耗链路带宽比率 (Consumed Bandwidth Ratio)
        CBR = 实际使用带宽 / 总带宽
        返回{(s1,s2): cbr_value}格式
        """
        cbr = {}
        for (node1, node2), traffic in self.traffic_data.items():
            # 检查链路带宽，考虑双向
            if (node1, node2) in self.link_bandwidths:
                bandwidth = self.link_bandwidths[(node1, node2)]
            elif (node2, node1) in self.link_bandwidths:
                bandwidth = self.link_bandwidths[(node2, node1)]
            else:
                raise ValueError(f"Bandwidth not defined for link ({node1}, {node2})")
                
            # 防止除以零错误
            if bandwidth == 0:
                cbr[(node1, node2)] = 0.0
            else:
                cbr[(node1, node2)] = traffic['total_bytes'] / bandwidth
        return cbr
    
    def calculate_alb(self):
        """
        计算可用链路带宽 (Available Link Bandwidth)
        ALB = 总带宽 - 实际使用带宽
        返回{(s1,s2): alb_value}格式
        """
        alb = {}
        for (node1, node2), traffic in self.traffic_data.items():
            # 检查链路带宽，考虑双向
            if (node1, node2) in self.link_bandwidths:
                bandwidth = self.link_bandwidths[(node1, node2)]
            elif (node2, node1) in self.link_bandwidths:
                bandwidth = self.link_bandwidths[(node2, node1)]
            else:
                raise ValueError(f"Bandwidth not defined for link ({node1}, {node2})")
                
            alb[(node1, node2)] = (bandwidth - traffic['total_bytes'])*8e-6
        return alb
    
    def get_all_metrics(self):
        """
        计算并返回所有链路指标
        返回格式: {(s1,s2): {'ebc': value, 'cbr': value, 'alb': value, 'll': 20}}
        """
        # 计算各指标
        ebc = self.calculate_ebc()
        cbr = self.calculate_cbr()  # 直接使用实例变量，无需传递参数
        alb = self.calculate_alb()  # 直接使用实例变量，无需传递参数
        
        # 整合所有指标，确保链路表示一致
        all_metrics = {}
        # 处理所有链路
        all_links = set(ebc.keys()).union(set(self.traffic_data.keys()))
        
        for link in all_links:
            node1, node2 = link
            # 确保能获取到所有指标
            current_ebc = ebc.get(link, ebc.get((node2, node1), 0.0))
            current_cbr = cbr.get(link, cbr.get((node2, node1), 0.0))
            current_alb = alb.get(link, alb.get((node2, node1), 0.0))
            
            all_metrics[(node1, node2)] = {
                'ebc': current_ebc,
                'cbr': current_cbr,
                'alb': current_alb,
                'll': self.ll
            }
            
        return all_metrics
    
    def write_dict_to_file(self):
        """
        将字典数据写入到指定的文件中
        """
        data=self.get_all_metrics()
        filename="/home/retr0/Project/TopologyObfu/CritiPro/output_file/link_metrics.txt"
        try:
            with open(filename, "w") as file:
                # 遍历字典，将每一项写入文件
                for key, value in data.items():
                    file.write(f"{key}: {value}\n")
            print(f"数据已成功写入文件 {filename}")
        except Exception as e:
            print(f"写入文件时发生错误: {e}")