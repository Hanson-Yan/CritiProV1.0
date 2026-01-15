# node_metrics.py
import networkx as nx
import threading
import time

"""
这里计算拓扑中每个节点的静态指标和动态指标，链路指标通过聚合的方式到相连接的节点上，
流量负载（aggregate_traffic_load）：直接求和。
带宽使用率（utilized_bandwidth_ratio）：计算所有链路的平均值。
可用带宽（available_link_bandwidth）：计算所有链路的平均值。
延迟（link_delay）：取最大值，反映最差情况。
"""


class NodeMetrics:
    def __init__(self, net, topo_matrix, traffic_data,bw_host_switch):
        self.net = net
        self.topo_matrix = topo_matrix
        self.graph = self._create_graph()
        self.traffic_data = traffic_data  # 从外部传入的流量数据
        self.static_metrics = self._calculate_static_metrics()
        self.dynamic_metrics = self._initialize_dynamic_metrics()
        self.update_interval = 5  # 动态指标更新间隔（秒）
        self.bw_host_switch=bw_host_switch

    def _create_graph(self):
        """根据 TopoMatrix 创建图结构"""
        graph = nx.Graph()
        for i in range(self.topo_matrix.n):
            for j in range(self.topo_matrix.n):
                if self.topo_matrix.matrix[i][j] is not None:
                    link = self.topo_matrix.matrix[i][j]
                    graph.add_edge(f's{i}', f's{j}', 
                                   bw=link.bw,
                                   delay=link.delay,
                                   loss=link.loss,
                                   max_queue_size=link.max_queue_size)
        return graph

    def _calculate_static_metrics(self):
        """计算静态指标"""
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)
        degree = nx.degree_centrality(self.graph)
        edge_betweenness = nx.edge_betweenness_centrality(self.graph)

        # 将最小割中心性聚合到节点上
        node_edge_betweenness = {node: 0 for node in self.graph.nodes}
        for (u, v), betweenness_value in edge_betweenness.items():
            node_edge_betweenness[u] += betweenness_value
            node_edge_betweenness[v] += betweenness_value

        return {
            'betweenness_centrality': betweenness,
            'closeness_centrality': closeness,
            'degree_centrality': degree,
            'node_edge_betweenness': node_edge_betweenness
        }

    def _initialize_dynamic_metrics(self):
        """初始化动态指标"""
        dynamic_metrics = {}
        for node in self.graph.nodes:
            dynamic_metrics[node] = {
                'aggregate_traffic_load': 0,
                'utilized_bandwidth_ratio': 0,
                'available_link_bandwidth': 0,
                'link_delay': 0
            }
        return dynamic_metrics

    def _calculate_dynamic_metrics(self):
        """
        计算动态指标
        动态指标的计算：
            首先是获取交换机所有端口的接收和转发流量，聚合流量负载为二者最大值，理论上接收流量等于转发流量
            交换机总带宽为所有链接的带宽相加，
            交换机链路利用率，为总负载除以总带宽后求平均
            交换机可用带宽，为总负载减去总带宽后求平均
            交换机链路延迟，所连接链路的最大延迟
            由于是只考虑关键节点，所以这样求是没有问题的，如果是考虑关键链路的话，应该要考虑交换机端口对应每条链路的负载和利用率
        """
        for node in self.graph.nodes:
            links = self.graph.edges(node, data=True)
            total_traffic_load = 0
            total_utilized_bw_ratio = 0
            total_available_bw = 0
            total_bw=0
            max_delay = 0
            
            for _, neighbor, link in links:
                bw = link['bw']
                delay = float(link['delay'].replace('ms', ''))  # 去掉单位
                total_bw+=bw
                max_delay = max(max_delay, delay)

            # 获取node对应的字典
            # print(f"node {node} links {len(links)} total_bw {total_bw}")
            node_data = self.traffic_data.get(node, {})
            bw_host_node_info=self.bw_host_switch.get(node, {'host_count': 0, 'total_bw': 0})
            #计入主机接入带宽
            host_node_bw=bw_host_node_info['total_bw']
            total_bw+=host_node_bw
            #计入主机接入链路
            links_num=len(links)+bw_host_node_info['host_count']

            # print(self.traffic_data)
            total_traffic_load=node_data['rx_bytes']
            total_utilized_bw_ratio=(total_traffic_load*8)/(total_bw*1000000)
            total_available_bw=total_bw-(total_traffic_load*8/1000000)
            
            #动态指标计算
            self.dynamic_metrics[node]['aggregate_traffic_load'] = total_traffic_load
            self.dynamic_metrics[node]['utilized_bandwidth_ratio'] = total_utilized_bw_ratio / links_num
            self.dynamic_metrics[node]['available_link_bandwidth'] = total_available_bw / links_num
            self.dynamic_metrics[node]['link_delay'] = max_delay



    def _update_dynamic_metrics(self):
        """
        周期计算动态指标
        需要使用线程持续统计端口流量
        """
        while True:
            for node in self.graph.nodes:
                links = self.graph.edges(node, data=True)
                total_traffic_load = 0
                total_utilized_bw_ratio = 0
                total_available_bw = 0
                total_bw=0
                max_delay = 0
                
                for _, neighbor, link in links:
                    bw = link['bw']
                    delay = float(link['delay'].replace('ms', ''))  # 去掉单位

                    # 获取链路的流量负载
                    # traffic_load = self.traffic_data
                    # utilized_bw_ratio = traffic_load / bw
                    # available_bw = bw - traffic_load

                    # total_traffic_load += traffic_load
                    # total_utilized_bw_ratio += utilized_bw_ratio
                    # total_available_bw += available_bw
                    total_bw+=bw
                    max_delay = max(max_delay, delay)

                # 获取node对应的字典
                # print(f"node {node} links {len(links)} total_bw {total_bw}")
                node_data = self.traffic_data.get(node, {})
                bw_host_node_info=self.bw_host_switch.get(node, {'host_count': 0, 'total_bw': 0})
                #计入主机接入带宽
                host_node_bw=bw_host_node_info['total_bw']
                total_bw+=host_node_bw
                #计入主机接入链路
                links_num=len(links)+bw_host_node_info['host_count']

                # print(self.traffic_data)
                total_traffic_load=node_data['rx_bytes']
                total_utilized_bw_ratio=(total_traffic_load*8)/(total_bw*1000000)
                total_available_bw=total_bw-(total_traffic_load*8/1000000)
                
                #动态指标计算
                self.dynamic_metrics[node]['aggregate_traffic_load'] = total_traffic_load
                self.dynamic_metrics[node]['utilized_bandwidth_ratio'] = total_utilized_bw_ratio / links_num
                self.dynamic_metrics[node]['available_link_bandwidth'] = total_available_bw / links_num
                self.dynamic_metrics[node]['link_delay'] = max_delay

                # 调试信息：打印每个节点的流量负载
                # print(f"Node {node}, Traffic Load: {total_traffic_load}, Utilized BW Ratio: {total_utilized_bw_ratio}, Available BW: {total_available_bw}, Delay: {max_delay}")

            time.sleep(self.update_interval)
            


    def get_node_metrics(self, node):
        """获取某个节点的所有指标"""
        if node not in self.graph.nodes:
            raise ValueError(f"节点 {node} 不存在于图中")
        
        static_metrics = {
            'betweenness_centrality': self.static_metrics['betweenness_centrality'][node],
            'closeness_centrality': self.static_metrics['closeness_centrality'][node],
            'node_edge_betweenness': self.static_metrics['node_edge_betweenness'][node],
            'degree_centrality': self.static_metrics['degree_centrality'][node]
        }
        
        dynamic_metrics = self.dynamic_metrics[node]
        
        return {**static_metrics, **dynamic_metrics}

    def get_all_node_metrics(self):
        """获取所有节点的指标"""
        all_metrics = {}
        for node in self.graph.nodes:
            all_metrics[node] = self.get_node_metrics(node)
        return all_metrics

    def start_dynamic_metrics_update(self):
        """启动动态指标更新线程"""
        # thread = threading.Thread(target=self._update_dynamic_metrics)
        # thread.daemon = True
        # thread.start()
        """静态计算动态指标"""
        self._calculate_dynamic_metrics()
    
    def write_dict_to_file(self):
        """
        将字典数据写入到指定的文件中
        """
        data=self.get_all_node_metrics()
        filename="/home/retr0/Project/TopologyObfu/CritiPro/output_file/metrics.txt"
        try:
            with open(filename, "w") as file:
                # 遍历字典，将每一项写入文件
                for key, value in data.items():
                    file.write(f"{key}: {value}\n")
            print(f"数据已成功写入文件 {filename}")
        except Exception as e:
            print(f"写入文件时发生错误: {e}")