from mininet.net import Mininet
# from mininet.node import Controller
from mininet.node import RemoteController
from mininet.node import OVSKernelSwitch,Host
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.node import CPULimitedHost
from mininet.util import quietRun
import TopoInput
import random
import time
import threading
import subprocess
import concurrent.futures
from mininet.node import OVSSwitch
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

import os
import signal

import sys
sys.path.append('/home/retr0/Project/TopologyObfu/CritiPro/critinode_model/')  # 修改为CritiPro目录的实际路径
from node_metrics import NodeMetrics
from link_metrics import LinkMetricsCalculator

# 导入关键节点识别模块
from critical_node_search import identify_key_nodes,identify_key_nodes_adaptive

from flood_test import run_test



def cleanup_mininet(signal, frame):
    """信号处理函数：执行sudo mn -c清理命令"""
    print("\n检测到Ctrl+C，正在清理Mininet环境...")
    try:
        # 执行sudo mn -c命令
        result = subprocess.run(
            ["sudo", "mn", "-c"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("清理完成：", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"清理失败：{e.stderr}")
    finally:
        # 退出程序
        exit(0)
signal.signal(signal.SIGINT, cleanup_mininet)



#主机接入带宽
host_link_bindwidth=100
#创建网络拓扑
class CustomTopo(Topo):
    def build(self,host_n=2,switch_n=1,filePath = None,edge_switch=[]):
        
        host_list=[]
        switch_list=[]
        # 添加主机
        for h in range(host_n):
            host=self.addHost('h%s'%h)
            host_list.append(host)

        # 添加交换机
        for s in range(switch_n):
            swtich=self.addSwitch('s%s'%s,protocols='OpenFlow13',stp=True)
            switch_list.append(swtich)
        
        print(host_list)
        print(switch_list)


        #利用路由矩阵添加交换机链路
        self.topoMatrix=TopoInput.switchTopoCreator.creatSwitchTopo(switch_n, filePath)
        for i in range(switch_n):
            for j in range(i,switch_n):
                if self.topoMatrix.matrix[i][j] is not None:
                    self.addLink(switch_list[i], switch_list[j], 
                                 bw=self.topoMatrix.matrix[i][j].bw, 
                                 delay=self.topoMatrix.matrix[i][j].delay, 
                                 loss=self.topoMatrix.matrix[i][j].loss, 
                                 max_queue_size=self.topoMatrix.matrix[i][j].max_queue_size, 
                                 use_htb=True)
                    # self.addLink(switch_list[i], switch_list[j])
        #根据输入的探测路由器序号列表添加主机和交换机的链路
        host_index=0
        for sw in edge_switch:
            # self.addLink(host_list[host_index],switch_list[sw],bw=10,delay='5ms', loss=0, max_queue_size=100, use_htb=True)
            self.addLink(host_list[host_index],switch_list[sw],bw=host_link_bindwidth,delay='20ms', loss=0, max_queue_size=100000, use_htb=True)
            host_index+=1

    def get_topo_matrix(self):
        return self.topoMatrix

       

# 挂载 local 环境到虚拟主机
def mount_local(net, local_path, mount_point):
    for host in net.hosts:
        # 创建挂载点目录
        host.cmd(f"mkdir -p {mount_point}")
        # 挂载宿主机的 Anaconda 环境到虚拟主机
        host.cmd(f"sudo mount --bind {local_path} {mount_point}")

# 卸载 Anaconda 环境并清理挂载点
def unmount_local(net, mount_point):
    for host in net.hosts:
        # 卸载挂载点
        host.cmd(f"sudo umount {mount_point}")
        # 删除挂载点目录
        host.cmd(f"rmdir {mount_point}")
        

# 流量生成任务
def generate_traffic_task(sender, receiver, port, bandwidth_str,interval):
    sender.cmd(f'sudo iperf -u -c {receiver.IP()} -p {port} -b {bandwidth_str} -t {interval} > /dev/null 2>&1 &')
     #流量负载记录
    link_key = (sender.name, receiver.name)
    # print(f"Recorded traffic for {link_key}: {traffic_data[link_key]}")  # 调试信息

# 使用线程池管理流量生成任务
def generate_continue_traffic(net, max_workers=3):
    port_base = 5001
    port_pool = list(range(port_base, port_base + 100))  # 创建端口池
    random.shuffle(port_pool)  # 打乱端口顺序

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            sender = random.choice(net.hosts)
            receiver = random.choice([h for h in net.hosts if h != sender])
            bandwidth = round(random.uniform(1, 5),2)  # 带宽范围 1-5M
            bandwidth_str = f"{bandwidth}M"
            interval = random.uniform(20, 30)  # 间隔时间 20-30秒
            port = port_pool.pop(0)  # 从端口池中取出一个端口

            executor.submit(generate_traffic_task, sender, receiver, port, bandwidth_str,interval)
            port_pool.append(port)  # 将端口放回端口池

            time.sleep(1)  # 控制任务提交的频率

def run_iperf_client(src, dst_ip, duration=10, bw="10M", udp=False):
    """
    启动 iperf 客户端
    :param src: Mininet host 对象
    :param dst_ip: 目标主机 IP
    :param duration: 持续时间 (秒)
    :param bw: 带宽限制 (如 '10M')
    :param udp: 是否使用 UDP 模式
    """
    udp_flag = "-u" if udp else ""
    cmd = f"iperf -c {dst_ip} -t {duration} -b {bw} {udp_flag}"
    print(f"[DEBUG] {src.name} -> {dst_ip} start: {cmd}")
    src.cmd(cmd)


def generate_background_traffic(net, duration=20, flow_count=5):
    """
    生成背景流量 (多进程版本)
    :param net: Mininet 网络对象
    :param duration: 每条流的持续时间
    :param flow_count: 同时运行的流数量
    """
    hosts = net.hosts
    procs = []

    flow_count = max(2, len(hosts)//2)

    for _ in range(flow_count):
        # 随机选择源宿主机
        src, dst = random.sample(hosts, 2)
        dst_ip = dst.IP()

        # 随机决定 TCP/UDP、带宽
        udp = random.choice([True, False])
        bw = random.choice(["1M", "5M", "10M", "20M"])

        # 在目标主机上启动 iperf server (后台运行)
        dst.cmd("pkill -9 iperf")  # 清理旧进程
        dst.cmd("iperf -s -p 5001 &")

        # 用 multiprocessing 启动客户端
        p = multiprocessing.Process(
            target=run_iperf_client, args=(src, dst_ip, duration, bw, udp)
        )
        procs.append(p)
        p.start()

        time.sleep(0.5)  # 避免所有流同时启动，模拟更自然

    # 等待所有进程完成
    # for p in procs:
    #     p.join()

    # print("[INFO] 背景流量生成结束。")
    return procs

#持续生成流量负载
def add_link_load(net,rate):
    h0=net.get('h0')
    for host in net.hosts:
        if host.name!="h0":
            h0.cmd(f"sudo iperf -c {host.IP()} -u -p 5001 -b {rate}M -t 3600 &")

#所有接受对开启探测接收程序
def start_probe_recive(net, probe_num):
    """
    这里注意脚本位置和映射的地址
    """
    for host in net.hosts:
        if host.name!="h0":
            host.cmd(f"sudo /py/python3 /probe/tpr.py {host.name}-eth0 {probe_num} &")
    print("Waiting for the probe receiver to start...")
    time.sleep(2)  # 等待探测程序启动

#开启探测发送程序
def start_probe_send(net, all_recevies_num, probe_num):
    time.sleep(1)
    print("Waiting for the probe sender to execute...")
    h0=net.get('h0')
    for num in range(all_recevies_num):
        if num == all_recevies_num-1:
            break
        recevies_num = num + 1
        h0.cmd(f"sudo /py/python3 /probe/tps.py {all_recevies_num} {recevies_num} {probe_num}")
    print("finish probe...")
    time.sleep(1)

def verify_pingall(net, max_attempts=3, retry_delay=1):
    """
    在已启动的Mininet网络中执行pingall并验证是否0%丢包
    
    参数:
        net: 已启动的Mininet网络对象
        max_attempts: 最大尝试次数，默认3次
        retry_delay: 重试前的等待时间(秒)，默认10秒
    
    返回:
        bool: 检查成功返回True，失败返回False
    """
    if not net or not hasattr(net, 'pingAll'):
        print("[ERROR] 无效的Mininet网络对象")
        return False
    
    print("[INFO] Running pingall to verify connectivity...")
    attempt = 0
    success = False
    
    while attempt < max_attempts and not success:
        attempt += 1
        # 执行pingall，返回丢包率
        loss = net.pingAll()
        
        # 打印pingall结果（模拟Mininet命令行输出格式）
        # hosts = net.hosts
        # print("*** Ping: testing ping reachability")
        # for src in hosts:
        #     reachable = []
        #     for dst in hosts:
        #         if src != dst:
        #             reachable.append(dst.name)
        #     print(f"{src.name} -> {' '.join(reachable)}")
        
        # total = len(hosts) * (len(hosts) - 1)
        # if total == 0:
        #     print("[WARNING] 没有可测试的主机连接")
        #     return True
            
        # received = total - int(loss * total / 100)
        # print(f"*** Results: {loss}% dropped ({received}/{total} received)")
        
        # 检查是否0%丢包
        if loss == 0:
            print("[INFO] All hosts are reachable (0% packet loss)")
            success = True
        else:
            print(f"[WARNING] Packet loss detected: {loss}%. Attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                # print(f"[INFO] Retrying after {retry_delay} seconds...")
                # time.sleep(retry_delay)
                continue
    
    if not success:
        print("[ERROR] Failed to achieve 0% packet loss after maximum attempts")
    
    return success


#获取网路中交换机的连接情况
def get_switch_port_connections(net,host_link_bw):
    """
    端节点连接了主机，动态指标中聚合流量考虑主机汇入的流量，但是没有考虑对应主机链路的带宽和利用率
    所以获取网路中交换机的连接情况，对连接了主机的交换机进行标记和计数，连接了多少台主机就计数多少，写入字典然后返回这些信息
    在考虑总带宽时，将连接了主机的交换机对应的主机带宽进行相加（规定所有接入主机的带宽是相同的）
    """
    # 初始化一个字典来存储每个交换机连接的主机数量
    switch_host_info = {}
    # 获取网络中的所有交换机
    switches = net.switches
    
    for switch in switches:
        
        switch_name = switch.name
        host_count=0
        total_host_link_bw=0
        # print(f"交换机 {switch_name} 的端口连接情况：")
        # 获取交换机的所有接口
        ports = switch.intfList()
        for port in ports:
            # 获取接口的连接信息
            link = port.link
            if link:
                # link是一个Link对象，包含两个接口对象
                src_intf, dst_intf = link.intf1, link.intf2
                # print(f"端口 {src_intf.name} 连接到 {dst_intf.name}")
                # 获取连接的另一端接口
                remote_intf = link.intf1 if link.intf1 != port else link.intf2
                # 检查连接的另一端是否是主机
                if isinstance(remote_intf.node, Host):
                    host_count += 1
                    
            # else:
                # print(f"端口 {port.name} 未连接")
        total_host_link_bw=host_link_bw*host_count
        # # 将交换机及其连接的主机数量存储到字典中
        # switch_host_count[switch_name] = total_host_link_bw
         # 将交换机及其连接的主机数量和总带宽存储到字典中
        switch_host_info[switch_name] = {'host_count': host_count, 'total_bw': total_host_link_bw}
        print(f"交换机 {switch_name} 连接的主机数量：{host_count}, 总带宽：{total_host_link_bw} Mbps")
    
    return switch_host_info

def get_link_bandwidths(net):
    """
    从 Mininet 网络中提取所有链路的带宽
    :param net: Mininet 网络实例
    :return: 字典，格式为 {(节点1, 节点2): 带宽值（字节/秒）}
    """
    link_bandwidths = {}
    for link in net.links:
        node1 = link.intf1.node.name
        node2 = link.intf2.node.name

        # TCLink 把参数分别存在 intf1.params 和 intf2.params
        bw1 = link.intf1.params.get('bw', None)
        bw2 = link.intf2.params.get('bw', None)

        # 如果两端都没设置，就设默认值，比如 100 Mbps
        bandwidth_mbps = bw1 or bw2 or 100  
        print(f"bandwidth_mbps {bandwidth_mbps}")
        bandwidth_bytes_per_sec = bandwidth_mbps * 125000

        if (node1, node2) not in link_bandwidths and (node2, node1) not in link_bandwidths:
            link_bandwidths[(node1, node2)] = bandwidth_bytes_per_sec

    return link_bandwidths



#使用Open vSwitch工具统计交换机端口流量信息
def get_switch_port_traffic_stats(port_name):
    """
    使用 ovs-vsctl 命令获取指定交换机端口的流量统计信息。
    :param port_name: 端口名称，例如 's1-eth1'
    :return: 字典，包含接收和发送的字节数
    """
    # 构建 ovs-vsctl 命令
    # print(f"execute sudo ovs-vsctl get Interface {port_name} statistics...")
    cmd = f"sudo ovs-vsctl get Interface {port_name} statistics"
    try:
        # 执行命令并捕获输出
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return {}

    # 解析输出结果，提取流量统计信息
    stats = {'rx_bytes': 0, 'tx_bytes': 0}
    # print("Raw output:", output.decode('utf-8'))  # 打印原始输出
    output=output.decode('utf-8')
    output = output.replace("\n", "").strip("{}")

    # 将字符串解析为字典
    parsed_dict = {}
    for item in output.split(", "):
        key, value = item.split("=")
        parsed_dict[key] = int(value)  # 假设所有值都是整数

    # 保存到 stats 字典中
    stats['rx_bytes'] = parsed_dict.get("rx_bytes", 0)
    stats['tx_bytes'] = parsed_dict.get("tx_bytes", 0)

    # 输出结果
    # print(stats)
    return stats

def get_switch_traffic_aggregation(net,traffic_time):
    """
    遍历所有交换机和端口，聚合流量负载。
    :param net: Mininet 网络实例
    :return: 字典，包含每个交换机的聚合流量负载
    """
    print("waiting for getting switch traffic aggregation...")
    time.sleep(traffic_time)
    traffic_aggregation = {}
    for switch in net.switches:
        switch_name = switch.name
        # 获取交换机的所有端口
        ports = switch.ports  # 直接使用 switch.ports 属性获取端口对象列表
        total_rx_bytes = 0
        total_tx_bytes = 0
        # print(f"ports {ports} swtich_name {switch_name}")
        for port_name, port in ports.items():
            port_name = f"{port_name}"  # 构建端口全名，例如 's1-eth1'
            if port_name!="lo":
                stats = get_switch_port_traffic_stats(port_name)
                total_rx_bytes += stats.get('rx_bytes', 0)
                total_tx_bytes += stats.get('tx_bytes', 0)
        traffic_aggregation[switch_name] = {'rx_bytes': total_rx_bytes/traffic_time, 'tx_bytes': total_tx_bytes/traffic_time}
    return traffic_aggregation

def get_link_traffic_data(net, traffic_time):
    """
    遍历所有链路，获取每条链路的流量数据。
    :param net: Mininet 网络实例
    :param traffic_time: 流量统计时长
    :return: 字典，包含每条链路的流量数据，键为链路(s1,s2)形式
    """
    print("waiting for getting link traffic data...")
    time.sleep(traffic_time)
    
    # 存储每个端口的流量数据
    port_traffic = {}
    
    # 首先收集所有交换机端口的流量数据
    for switch in net.switches:
        switch_name = switch.name
        ports = switch.ports  # 获取交换机的所有端口
        
        for port_name, port in ports.items():
            full_port_name = f"{port_name}"  # 端口全名，如 's1-eth1'
            if full_port_name != "lo":  # 忽略回环接口
                stats = get_switch_port_traffic_stats(full_port_name)
                port_traffic[(switch_name, full_port_name)] = {
                    'rx_bytes': stats.get('rx_bytes', 0) / traffic_time,
                    'tx_bytes': stats.get('tx_bytes', 0) / traffic_time
                }
    
    # 关联链路与端口流量数据
    link_traffic = {}
    for link in net.links:
        # 获取链路两端的交换机和端口
        node1 = link.intf1.node.name
        node2 = link.intf2.node.name
        port1 = link.intf1.name
        port2 = link.intf2.name
        
        # 从端口流量数据中获取对应的值
        traffic1 = port_traffic.get((node1, port1), {'rx_bytes': 0, 'tx_bytes': 0})
        traffic2 = port_traffic.get((node2, port2), {'rx_bytes': 0, 'tx_bytes': 0})
        
        # 计算链路的总流量（双向流量之和的平均值）
        total_traffic = (traffic1['tx_bytes'] + traffic2['tx_bytes'] + 
                         traffic1['rx_bytes'] + traffic2['rx_bytes']) / 2
        
        # 以(s1,s2)形式存储链路流量数据
        link_traffic[(node1, node2)] = {
            'rx_bytes': (traffic1['rx_bytes'] + traffic2['rx_bytes']) / 2,
            'tx_bytes': (traffic1['tx_bytes'] + traffic2['tx_bytes']) / 2,
            'total_bytes': total_traffic
        }
    
    return link_traffic

def draw_mininet_topology(net,save_or_no):
    """
    在单独的进程中绘制 Mininet 网络拓扑图（无向图）
    :param net: Mininet 网络对象
    """
    save_path = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/topo_original.png"
    # 创建一个无向图
    G = nx.Graph()

    # 添加主机
    hosts = net.hosts
    for host in hosts:
        G.add_node(host.name, color='lightblue', shape='s')  # 主机用蓝色正方形表示

    # 添加交换机
    switches = net.switches
    for switch in switches:
        G.add_node(switch.name, color='lightgreen', shape='o')  # 交换机用绿色圆形表示

    # 添加链路
    links = net.links
    for link in links:
        src = link.intf1.node.name
        dst = link.intf2.node.name
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)

    # 设置图形布局
    # pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
    pos = nx.kamada_kawai_layout(G)
    node_size = 200
    font_size = 8
    # 绘制节点
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    node_shapes = [data['shape'] for _, data in G.nodes(data=True)]
    for shape in set(node_shapes):
        nodes = [node for node, data in G.nodes(data=True) if data['shape'] == shape]
        # nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[G.nodes[node]['color'] for node in nodes], node_shape=shape)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[G.nodes[node]['color'] for node in nodes], node_shape=shape,node_size=node_size)

    # 绘制边
    nx.draw_networkx_edges(G, pos)

    # 添加标签
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_labels(G, pos,font_size=font_size)

    # 设置图形样式
    plt.style.use('default')  # 使用默认样式
    plt.box(False)  # 关闭边框

    # 显示图形
    plt.title("Mininet Network Topology")
    plt.axis('off')  # 关闭坐标轴
    if save_or_no:
        plt.savefig(save_path, format='png', dpi=600)
        print(f"topo original have saved in \n{save_path} ")
    plt.show()

# 绘制网络拓扑图并突出显示关键节点
def draw_mininet_topology_with_critical_nodes(net, critical_nodes):
    """
    在单独的进程中绘制 Mininet 网络拓扑图，并突出显示关键节点。
    :param net: Mininet 网络实例
    :param critical_nodes: 关键节点列表
    """
    save_path="/home/retr0/Project/TopologyObfu/CritiPro/output_file/topo_original_critical.png"
    # 创建一个无向图
    G = nx.Graph()

    # 添加主机
    hosts = net.hosts
    for host in hosts:
        G.add_node(host.name, color='lightblue', shape='s')  # 主机用蓝色正方形表示

    # 添加交换机
    switches = net.switches
    for switch in switches:
        G.add_node(switch.name, color='lightgreen', shape='o')  # 交换机用绿色圆形表示

    # 添加链路
    links = net.links
    for link in links:
        src = link.intf1.node.name
        dst = link.intf2.node.name
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)

    # 设置图形布局
    # pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
    pos = nx.kamada_kawai_layout(G)
    node_size = 200
    font_size = 8

    # 绘制节点
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    node_shapes = [data['shape'] for _, data in G.nodes(data=True)]
    for shape in set(node_shapes):
        nodes = [node for node, data in G.nodes(data=True) if data['shape'] == shape]
        # nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[G.nodes[node]['color'] for node in nodes], node_shape=shape)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[G.nodes[node]['color'] for node in nodes], node_shape=shape,node_size=node_size)

    # 突出显示关键节点
    critical_node_names = [node for node, _ in critical_nodes]
    critical_node_colors = ['red'] * len(critical_node_names)
    nx.draw_networkx_nodes(G, pos, nodelist=critical_node_names, node_color=critical_node_colors, node_shape='o', node_size=250, edgecolors='black', linewidths=2)

    # 绘制边
    nx.draw_networkx_edges(G, pos)

    # 添加标签
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_labels(G, pos,font_size=font_size)

    # 设置图形样式
    plt.style.use('default')  # 使用默认样式
    plt.box(False)  # 关闭边框

    # 显示图形
    plt.title("Original Network Topology with Critical Nodes")
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, format='png', dpi=600)
    print(f"originial topo with ctiticla nodes have saved in\n{save_path}")
    plt.show()


# def run_probe(topo_num):
#     host_n, switch_n,  edge_switch= get_topo_info(topo_num)
#     topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/"

#     # file_name = input("请输入拓扑文件名 : ")
#     # host_n = int(input("请输入主机数量 : "))
#     # switch_n = int(input("请输入交换机数量 : "))
#     # edge_switch = list(map(int, input("请按顺序输入连接的节点编号（从0开始,用空格分隔）: ").split()))
#     file_path = topo_matrix_path + topo_num +".txt"
#     if not os.path.exists(file_path):
#         print(f"文件 {file_path} 不存在")
#         sys.exit(1)
#     topo = CustomTopo(host_n,switch_n,filePath=file_path,edge_switch=edge_switch)
#     # net = Mininet(topo=topo,host=CPULimitedHost , link=TCLink, autoStaticArp=True,switch=OVSSwitch,autoSetMacs=True)
#     controller = RemoteController('c0', ip='127.0.0.1', port=6633)
#     net = Mininet(topo=topo,
#               controller=controller,
#               host=CPULimitedHost,
#               link=TCLink,
#               switch=partial(OVSSwitch, protocols='OpenFlow13',stp=True))

    
#     # 启动网络
#     net.start()
#     print("Mininet is running. You can now test the network.")

#     #绘制网络拓扑图
#     # 使用 multiprocessing 创建一个子进程进行绘图
#     drawing_process = multiprocessing.Process(target=draw_mininet_topology, args=(net,0))
#     drawing_process.start()
#     time.sleep(40)

#     # 宿主机的 Anaconda 环境路径\探测脚本路径
#     anaconda_local_path = "/home/retr0/anaconda3/bin/"
#     anaconda_mount_point = "/py"
#     probe_local_path = "/home/retr0/Project/TopologyObfu/MininetTop/probeCode"
#     probe_mount_point = "/probe"
    
#     # 为每个主机挂载 Anaconda 环境和探测脚本
#     mount_local(net,anaconda_local_path,anaconda_mount_point)
#     mount_local(net,probe_local_path,probe_mount_point)


#     # 进入 Mininet CLI
#     #  终止绘图进程
#     drawing_process.terminate()
#     drawing_process.join()  # 确保绘图进程已终止
#     # add_link_load(net,2)
#     CLI(net)

#     #卸载挂载点
#     unmount_local(net,anaconda_mount_point)
#     unmount_local(net,probe_mount_point)
    
#     # 停止网络
#     net.stop()
    
def collect_link_metric(topo_num):
# 创建拓扑
    topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"

    host_n, switch_n,  edge_switch= get_topo_info(topo_num)
    file_path = topo_matrix_path + topo_num +".txt"
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        sys.exit(1)
    topo = CustomTopo(host_n,switch_n,filePath=file_path,edge_switch=edge_switch)
    net = Mininet(topo=topo,host=CPULimitedHost , link=TCLink, autoStaticArp=True,switch=OVSSwitch,autoSetMacs=True)

    # 启动网络
    net.start()
    print("Mininet is running. You can now test the network.")

    traffic_thread = threading.Thread(target=generate_continue_traffic, args=(net,))
    traffic_thread.daemon = True  # 设置为守护线程，这样主线程退出时，流量生成线程也会退出
    traffic_thread.start()

    # 统计交换机端口流量信息
    traffic_link_data = {}
    traffic_time=20
    traffic_link_data=get_link_traffic_data(net,traffic_time)
    print(f"traffic_link_data {traffic_link_data}")
    link_bindwidths = get_link_bandwidths(net)

    # 创建 NodeMetrics 实例
    link_metrics = LinkMetricsCalculator(net, link_bindwidths, traffic_link_data)
    link_metrics.write_dict_to_file()

    # 等待节点指标写入
    time.sleep(2)  
        # 停止网络
    net.stop()

def collect_node_metric(topo_num):
# 创建拓扑
    topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"

    host_n, switch_n,  edge_switch= get_topo_info(topo_num)
    file_path = topo_matrix_path + topo_num +".txt"
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        sys.exit(1)
    topo = CustomTopo(host_n,switch_n,filePath=file_path,edge_switch=edge_switch)
    net = Mininet(topo=topo,host=CPULimitedHost , link=TCLink, autoStaticArp=True,switch=OVSSwitch,autoSetMacs=True)
    
    # 启动网络
    net.start()
    print("Mininet is running. You can now test the network.")

    traffic_thread = threading.Thread(target=generate_continue_traffic, args=(net,))
    traffic_thread.daemon = True  # 设置为守护线程，这样主线程退出时，流量生成线程也会退出
    traffic_thread.start()


    # 统计交换机端口流量信息
    traffic_data = {}
    traffic_time=20
    traffic_data=get_switch_traffic_aggregation(net,traffic_time)
    print(f"traffic_data {traffic_data}")
    #主机接入带宽统计
    bw_host_switch={}
    bw_host_switch=get_switch_port_connections(net,host_link_bindwidth)
    print(get_switch_port_connections(net,host_link_bindwidth))

    # 创建 NodeMetrics 实例
    node_metrics = NodeMetrics(net, topo.get_topo_matrix(), traffic_data,bw_host_switch)
    node_metrics.start_dynamic_metrics_update()

    # #写入指标
    # print("Metrics for all nodes:", node_metrics.get_all_node_metrics())
    node_metrics.write_dict_to_file()

    # 等待节点指标写入
    time.sleep(2)   
    
    # 停止网络
    net.stop()



def run_draw(topo_num):
# 创建拓扑
    topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"

    host_n, switch_n,  edge_switch= get_topo_info(topo_num)
    file_path = topo_matrix_path + topo_num +".txt"
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        sys.exit(1)
    topo = CustomTopo(host_n,switch_n,filePath=file_path,edge_switch=edge_switch)
    net = Mininet(topo=topo,host=CPULimitedHost , link=TCLink, autoStaticArp=True,switch=OVSSwitch,autoSetMacs=True)
    # controller = RemoteController('c0', ip='127.0.0.1', port=6633)
    # net = Mininet(topo=topo,
    #           controller=controller,
    #           host=CPULimitedHost,
    #           link=TCLink,
    #           switch=partial(OVSSwitch, protocols='OpenFlow13',stp=True))

    
    # 启动网络
    net.start()
    print("Mininet is running. You can now test the network.")

    #绘制网络拓扑图
    # 使用 multiprocessing 创建一个子进程进行绘图
    drawing_process = multiprocessing.Process(target=draw_mininet_topology, args=(net,1))
    drawing_process.start()

    # 宿主机的 Anaconda 环境路径\探测脚本路径
    anaconda_local_path = "/home/retr0/anaconda3/bin/"
    anaconda_mount_point = "/py"
    probe_local_path = "/home/retr0/Project/TopologyObfu/MininetTop/probeCode"
    probe_mount_point = "/probe"
    
    # 为每个主机挂载 Anaconda 环境和探测脚本
    mount_local(net,anaconda_local_path,anaconda_mount_point)
    mount_local(net,probe_local_path,probe_mount_point)

    # traffic_thread = threading.Thread(target=generate_continue_traffic, args=(net,))
    # traffic_thread.daemon = True  # 设置为守护线程，这样主线程退出时，流量生成线程也会退出
    # traffic_thread.start()

    # traffic_process = multiprocessing.Process(target=generate_background_traffic, args=(net, 30, 10))
    # traffic_process.start()
    procs = generate_background_traffic(net, duration=30, flow_count=10)

    # 统计交换机端口流量信息
    traffic_data = {}
    traffic_time=20
    traffic_data=get_switch_traffic_aggregation(net,traffic_time)
    print(f"traffic_data {traffic_data}")
    #主机接入带宽统计
    bw_host_switch={}
    bw_host_switch=get_switch_port_connections(net,host_link_bindwidth)
    print(get_switch_port_connections(net,host_link_bindwidth))

    # 创建 NodeMetrics 实例
    node_metrics = NodeMetrics(net, topo.get_topo_matrix(), traffic_data,bw_host_switch)
    node_metrics.start_dynamic_metrics_update()

    # #写入指标
    # print("Metrics for all nodes:", node_metrics.get_all_node_metrics())
    node_metrics.write_dict_to_file()

    # 等待节点指标写入
    time.sleep(2)  
    file_path = '/home/retr0/Project/TopologyObfu/CritiPro/output_file/metrics.txt'  # 关键节点识别程序的输入文件路径

    identify_start_time = time.perf_counter()
    # critical_nodes = identify_key_nodes(file_path)
    critical_nodes = identify_key_nodes_adaptive(file_path)
    identify_end_time = time.perf_counter()
    identify_duration_time = identify_end_time - identify_start_time
    print(f"关键节点识别时间为：{identify_duration_time:.6f} s")

    print("识别的关键节点如下：")
    node_num=[]
    for node, score in critical_nodes:
        print(f"节点 {node}: 得分 = {score:.4f}")
        numbers = int(''.join(filter(str.isdigit, node)))
        node_num.append(numbers)
    import numpy as np
    critical_nodes_save_path=f"/home/retr0/Project/TopologyObfu/CritiPro/output_file/critical_nodes.txt"
    np.savetxt(critical_nodes_save_path,node_num,fmt="%d")
    # 绘制网络拓扑图并突出显示关键节点
    critical_drawing_process = multiprocessing.Process(target=draw_mininet_topology_with_critical_nodes, args=(net, critical_nodes))
    critical_drawing_process.start()
    time.sleep(5)


    # 进入 Mininet CLI
    CLI(net)

    #卸载挂载点
    unmount_local(net,anaconda_mount_point)
    unmount_local(net,probe_mount_point)
    for p in procs:
        p.join()
    # 停止网络
    net.stop()
    #  终止绘图进程
    # traffic_process.terminate()
    # traffic_process.join()
    drawing_process.terminate()
    drawing_process.join()  # 确保绘图进程已终止
    critical_drawing_process.terminate()
    critical_drawing_process.join()

def progress_bar(total_time):
    # 总时间（秒）
    total_seconds = total_time
    # 进度条长度
    bar_length = 50

    for elapsed_time in range(total_seconds + 1):
        # 计算已完成的百分比
        progress = elapsed_time / total_seconds
        # 计算已完成的进度条长度
        bar_filled_length = int(round(bar_length * progress))
        # 创建进度条字符串
        bar = "#" * bar_filled_length + "-" * (bar_length - bar_filled_length)
        # 计算剩余时间
        remaining_time = total_seconds - elapsed_time
        # 格式化输出
        sys.stdout.write(f"\r|{bar}| {progress * 100:.2f}% Complete, Remaining: {remaining_time} seconds")
        sys.stdout.flush()
        # 暂停1秒
        time.sleep(1)

    print("\nProgress complete!")


# host_locks = {}  # 每个 host 一个锁

# def run_iperf_pair(h1, h2, port, duration):
#     lock1 = host_locks[h1.name]
#     lock2 = host_locks[h2.name]

#     with lock1, lock2:
#         h2.cmd(f'iperf -s -p {port} -u &')
#         time.sleep(0.5)
#         output = h1.cmd(f'iperf -c {h2.IP()} -p {port} -u -t {duration} -b 10M')
#         h2.cmd('kill %iperf')

#     # 解析吞吐量
#     throughput = 0.0
#     for line in output.strip().split('\n'):
#         if "Mbits/sec" in line:
#             try:
#                 throughput = float(line.split()[-2])
#             except:
#                 pass
#     print(f"[{h1.name} -> {h2.name}] throughput: {throughput} Mbits/sec")
#     return throughput


def measure_throughput(topo_num,duration=10):
    """
    在部署模型前，测量整个网络在随机流之间的吞吐量。
    """
    before_or_after=input(f"deploy?(before:1,after:2):")
    topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"

    host_n, switch_n,  edge_switch= get_topo_info(topo_num)
    file_path = topo_matrix_path + topo_num +".txt"
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        sys.exit(1)
    topo = CustomTopo(host_n,switch_n,filePath=file_path,edge_switch=edge_switch)
    # net = Mininet(topo=topo,host=CPULimitedHost , link=TCLink, autoStaticArp=True,switch=OVSSwitch,autoSetMacs=True)
    controller = RemoteController('c0', ip='127.0.0.1', port=6633)
    net = Mininet(topo=topo,
              controller=controller,
              host=CPULimitedHost,
              link=TCLink,
              switch=partial(OVSSwitch, protocols='OpenFlow13',stp=True))
    # 启动网络
    net.start()
    print("routing...")
    # time.sleep(60)
    progress_bar(60)


    hosts = net.hosts
    total_throughput = 0.0

    print("开始测量吞吐量...")
    if before_or_after=="2":
        if os.path.exists("/tmp/enable_delay_signal"):
            os.remove("/tmp/enable_delay_signal")
        with open("/tmp/enable_delay_signal", "w") as f:
            f.write("go")
        print("✅ 通知控制器激活延迟逻辑")
    # for i in range(len(hosts)):
    #     for j in range(len(hosts)):
    #         if i != j:
    #             h1 = hosts[i]
    #             h2 = hosts[j]
    #             port = random.randint(5000, 6000)

    #             # 启动 iperf server
    #             h2.cmd(f'iperf -s -p {port} -u &')
    #             time.sleep(0.5)  # 等待 server 启动

    #             # 启动 iperf client 发送 UDP 流量
    #             print(f"Testing {h1.name} -> {h2.name}")
    #             output = h1.cmd(f'iperf -c {h2.IP()} -p {port} -u -t {duration} -b 10M')
                
    #             # 解析吞吐量
    #             lines = output.strip().split('\n')
    #             for line in lines:
    #                 if "Mbits/sec" in line:
    #                     try:
    #                         throughput = float(line.split()[-2])
    #                         total_throughput += throughput
    #                     except:
    #                         pass

    #             # 杀死 server
    #             h2.cmd('kill %iperf')


    h0 = net.get('h0')
    total_throughput = 0.0

    print("开始测量 h0 -> 所有主机的吞吐量...")
    for host in net.hosts:
        if host == h0:
            continue
        port = random.randint(5000, 6000)

        # 启动 iperf server
        host.cmd(f'iperf -s -p {port} -u &')
        time.sleep(0.5)

        print(f"Testing h0 -> {host.name}")
        output = h0.cmd(f'iperf -c {host.IP()} -p {port} -u -t {duration} -b 10M')

        # 解析吞吐量
        for line in output.strip().split('\n'):
            if "Mbits/sec" in line:
                try:
                    throughput = float(line.split()[-2])
                    total_throughput += throughput
                except:
                    pass

        host.cmd('kill %iperf')

    # h0 = net.get('h0')
    # total_throughput = 0.0

    # print(f"开始测量 h0 -> 所有主机（端口 {port}）的吞吐量...")
    # for host in net.hosts:
    #     if host == h0:
    #         continue

    #     # 清除旧进程，避免端口冲突
    #     host.cmd('killall -9 iperf')
    #     h0.cmd('killall -9 iperf')
    #     time.sleep(0.2)

    #     # 启动服务端
    #     host.cmd(f'iperf -s -p {port} -u &')
    #     time.sleep(0.5)

    #     # 客户端发送流量
    #     print(f"Testing h0 -> {host.name}")
    #     output = h0.cmd(f'iperf -c {host.IP()} -p {port} -u -t {duration} -b 10M')

    #     # 解析吞吐量
    #     for line in output.strip().split('\n'):
    #         if "Mbits/sec" in line:
    #             try:
    #                 throughput = float(line.split()[-2])
    #                 total_throughput += throughput
    #             except:
    #                 pass

    #     host.cmd('kill %iperf')

    


        # 停止网络
    net.stop()
    if before_or_after=="1":
        file_name=f"/home/retr0/Project/TopologyObfu/Experiment/deploy_throughput/{topo_num}.txt"
        string_to_write=f"部署模型前,总吞吐量（UDP）：{total_throughput:.2f} Mbits/sec"
        print(string_to_write)
        with open(file_name, "a") as file:
            file.write(string_to_write + "\n")  # 写入字符串，并换行
    elif before_or_after=="2":
        file_name=f"/home/retr0/Project/TopologyObfu/Experiment/deploy_throughput/{topo_num}.txt"
        string_to_write=f"部署模型后,总吞吐量（UDP）：{total_throughput:.2f} Mbits/sec"
        print(string_to_write)
        with open(file_name, "a") as file:
            file.write(string_to_write + "\n")  # 写入字符串，并换行
        if os.path.exists("/tmp/enable_delay_signal"):
            os.remove("/tmp/enable_delay_signal")
            print("临时文件已清理")
    else:
        print(f"总吞吐量（UDP）：{total_throughput:.2f} Mbits/sec")
    
    return total_throughput

def cleanup_network(net):
    """清理网络"""
    print("[CLEANUP] 清理网络中...")
    for host in net.hosts:
        host.cmd('killall -9 iperf iperf3 2>/dev/null')
        host.cmd('pkill -9 -f iperf 2>/dev/null')
    time.sleep(2)
    print("[CLEANUP] ✅ 完成\n")

def measure_link_flood(topo_num):
    # 创建拓扑
    topo_matrix_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"

    host_n, switch_n, edge_switch = get_topo_info(topo_num)
    file_path = topo_matrix_path + topo_num + ".txt"
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        sys.exit(1)

    topo = CustomTopo(host_n, switch_n, filePath=file_path, edge_switch=edge_switch)
    controller = RemoteController('c0', ip='127.0.0.1', port=6633)
    net = Mininet(
        topo=topo,
        controller=controller,
        host=CPULimitedHost,
        link=TCLink,
        switch=partial(OVSSwitch, protocols="OpenFlow13", stp=True),
        autoSetMacs=True,
        autoStaticArp=True,
    )

    # 启动网络
    net.start()
    print("[INFO] Mininet started, waiting for routes to install...")
    verify_pingall(net,max_attempts=1)
    # ========== baseline 测试 ==========
    print("[TEST] Running baseline test...")
    baseline = run_test(net, "h2", "h5", attack=None, duration=10)
    print("Baseline:", baseline)
    
    cleanup_network(net)  # 👈 清理
    # ========== 攻击真实关键链路 ==========
    print("[TEST] Running attack on true critical link...")
    attack_true = run_test(net, "h2", "h5", attack=("h0", "h3"), duration=10)
    print("Attack (true critical):", attack_true)
    
    cleanup_network(net)  # 👈 清理
    # ========== baseline2 测试 ==========
    print("[TEST] Running baseline test2 (验证恢复)...")
    baseline2 = run_test(net, "h2", "h5", attack=None, duration=10)
    print("Baseline2:", baseline2)
    
    # 验证清理效果
    if baseline['throughput_Mbps'] and baseline2['throughput_Mbps']:
        diff = abs(baseline['throughput_Mbps'] - baseline2['throughput_Mbps']) / baseline['throughput_Mbps'] * 100
        if diff > 10:
            print(f"[WARN] ⚠️ Baseline 差异 {diff:.1f}%，可能清理不彻底")
        else:
            print(f"[INFO] ✅ Baseline 差异仅 {diff:.1f}%，网络已恢复")
    
    cleanup_network(net)  # 👈 清理
    # ========== 攻击混淆链路 ==========
    print("[TEST] Running attack on obfuscated link...")
    attack_fake = run_test(net, "h2", "h5", attack=("h1", "h0"), duration=10)
    print("Attack (obfuscated):", attack_fake)

    # # ========== baseline 测试 ==========
    # print("[TEST] Running baseline test...")
    # baseline = run_test(net, "h3", "h7", attack=None, duration=10)
    # print("Baseline:", baseline)

    # # ========== 攻击真实关键链路 ==========
    # print("[TEST] Running attack on true critical link...")
    # attack_true = run_test(net, "h3", "h7", attack=("h2", "h9"), duration=10)
    # print("Attack (true critical):", attack_true)

    # # ========== 攻击混淆链路 ==========
    # print("[TEST] Running attack on obfuscated link...")
    # attack_fake = run_test(net, "h3", "h7", attack=("h1", "h2"), duration=10)
    # print("Attack (obfuscated):", attack_fake)

    # 保存结果（可选：写 CSV）
    # import pandas as pd
    # results = pd.DataFrame(
    #     [baseline, attack_true, attack_fake],
    #     index=["baseline", "true_attack", "fake_attack"],
    # )
    # results.to_csv(f"/home/retr0/Project/TopologyObfu/Experiment/flood_test/results_{topo_num}.csv")
    # print(f"[INFO] Results saved to results_{topo_num}.csv")

    import json
    results = {
        "baseline": baseline,
        "true_attack": attack_true,
        "fake_attack": attack_fake
    }

    # 确保目录存在
    save_dir = "/home/retr0/Project/TopologyObfu/Experiment/flood_test/"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}results_{topo_num}.json", "w") as f:
        json.dump(results, f, indent=4)  # indent=4让JSON更易读
    print(f"[INFO] Results saved to {save_dir}results_{topo_num}.json")

    # CLI 可选：调试用
    # CLI(net)

    # 停止网络
    net.stop()
    print("[INFO] Mininet stopped.")



def input_topo_info():
    topo_matrix_dir="/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/"
    topo_num = input("请输入拓扑名（topo_num）:")
    topo_num_txt = topo_matrix_dir+topo_num+".txt"
    if not os.path.exists(topo_num_txt):
        print(f"{topo_num_txt} is not exist.please check.")
        sys.exit(1)
    # 提示用户输入信息

    host_num = int(input("请输入主机数量 (host_num): "))
    switch_num = int(input("请输入交换机数量 (switch_num): "))
    
    # 提示用户输入连接顺序（整数数组）
    while True:
        try:
            connect_switch_order = input("请输入连接顺序 (connect_switch_order)，用空格分隔整数: ")
            connect_switch_order = list(map(int, connect_switch_order.split()))
            break
        except ValueError:
            print("输入无效，请确保输入的是用空格分隔的整数！")

    # 将输入的信息存储到字典中
    user_data = {
        "host_num": host_num,
        "switch_num": switch_num,
        "connect_switch_order": connect_switch_order
    }

    # 将字典写入到文件中
    file_name = topo_matrix_dir+topo_num+"_info.txt"
    with open(file_name, "w") as file:
        for key, value in user_data.items():
            if isinstance(value, list):
                # 如果值是列表，将其转换为字符串形式
                file.write(f"{key}: {', '.join(map(str, value))}\n")
            else:
                file.write(f"{key}: {value}\n")

    print(f"用户输入已成功保存到文件 {file_name} 中！")

def get_topo_info(topo_num):
    topo_matrix = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"
    file_name = topo_matrix+topo_num+"_info.txt"
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

def check_topo_info():
    topo_matrix_dir="/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/"
    topo_num = input("请输入拓扑名（topo_num）:")
    topo_num_txt = topo_matrix_dir+topo_num+"_info.txt"
    if not os.path.exists(topo_num_txt):
        print(f"{topo_num_txt} is not exist.please input topo info firstly.")
        sys.exit(1)
    return topo_num


if __name__ == '__main__':
    topo_num=check_topo_info()
    # run_draw(topo_num)
    # measure_throughput(topo_num)
    # collect_link_metric(topo_num)
    # collect_node_metric(topo_num)
    measure_link_flood(topo_num)
    