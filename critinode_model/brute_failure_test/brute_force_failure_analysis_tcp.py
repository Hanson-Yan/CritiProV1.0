#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
暴力故障影响分析实验（分层采样版本）

核心改进：
1. ✓ 动态识别 h0 的接入交换机
2. ✓ 只对 h0 的接入交换机使用多对多测量
3. ✓ 使用分层采样避免估算偏差
4. ✓ 数学上无偏的估计量

作者: Retr0
日期: 2024
"""

import os
import sys
import time
import random
import signal
import subprocess
import csv
import json
import numpy as np
from functools import partial
from datetime import datetime

from mininet.net import Mininet
from mininet.node import RemoteController, CPULimitedHost, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo

# 导入关键节点识别模块
sys.path.append('/home/retr0/Project/TopologyObfu/CritiPro/critinode_model/')
from critical_node_search import identify_key_nodes_adaptive

# ==================== 配置参数 ====================
CONFIG = {
    'iperf_duration': 5,
    'iperf_bandwidth': '10M',
    'convergence_wait': 3,
    'recovery_wait': 3,
    'pingall_max_attempts': 2,
    'controller_ip': '127.0.0.1',
    'controller_port': 6633,
    'host_link_bandwidth': 100,
    'routing_convergence_time': 30,
    'baseline_repeat': 1,
    'failure_repeat': 1,
    'multi_pair_sample_ratio': 0.3,
    'significance_level': 0.05,
    'output_dir': '/home/retr0/Project/TopologyObfu/Experiment/brute_force_results/',
    'topo_base_path': '/home/retr0/Project/TopologyObfu/CritiPro/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==================== 工具函数 ====================

def cleanup_mininet(signal_num=None, frame=None):
    """清理 Mininet 环境"""
    print("\n[CLEANUP] 正在清理 Mininet 环境...")
    try:
        subprocess.run(["sudo", "mn", "-c"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[CLEANUP] 清理完成")
    except subprocess.CalledProcessError as e:
        print(f"[CLEANUP] 清理失败: {e}")
    finally:
        if signal_num is not None:
            exit(0)

signal.signal(signal.SIGINT, cleanup_mininet)

def progress_bar(current, total, prefix='Progress', bar_length=50):
    """显示进度条"""
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
    sys.stdout.flush()
    if current == total:
        print()

def log_message(message, level='INFO'):
    """格式化日志输出"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

# ==================== 拓扑加载 ====================

class CustomTopo(Topo):
    """自定义拓扑类"""
    def build(self, host_n=2, switch_n=1, filePath=None, edge_switch=[]):
        host_list = []
        switch_list = []
        
        for h in range(host_n):
            host = self.addHost(f'h{h}')
            host_list.append(host)
        
        for s in range(switch_n):
            switch = self.addSwitch(f's{s}', protocols='OpenFlow13', stp=False)
            switch_list.append(switch)
        
        log_message(f"创建拓扑: {host_n} 主机, {switch_n} 交换机")
        
        self.topoMatrix = self._load_topo_matrix(switch_n, filePath)
        for i in range(switch_n):
            for j in range(i, switch_n):
                if self.topoMatrix[i][j] == 1:
                    self.addLink(switch_list[i], switch_list[j], 
                               bw=100, delay='10ms', loss=0, 
                               max_queue_size=10000, use_htb=True)
        
        for host_index, sw_index in enumerate(edge_switch):
            self.addLink(host_list[host_index], switch_list[sw_index],
                        bw=CONFIG['host_link_bandwidth'], delay='20ms', 
                        loss=0, max_queue_size=100000, use_htb=True)
        
        log_message(f"主机连接配置: {list(zip([f'h{i}' for i in range(host_n)], [f's{i}' for i in edge_switch]))}")
    
    def _load_topo_matrix(self, switch_n, filePath):
        """加载拓扑邻接矩阵"""
        matrix = [[0] * switch_n for _ in range(switch_n)]
        try:
            with open(filePath, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i >= switch_n:
                        break
                    values = list(map(int, line.strip().split()))
                    for j, val in enumerate(values):
                        if j >= switch_n:
                            break
                        matrix[i][j] = val
        except Exception as e:
            log_message(f"加载拓扑矩阵失败: {e}", 'ERROR')
            sys.exit(1)
        return matrix
    
    def get_topo_matrix(self):
        return self.topoMatrix

def get_topo_info(topo_num):
    """读取拓扑配置信息"""
    info_file = os.path.join(CONFIG['topo_base_path'], 
                             f'topo_{topo_num}_result/topo_{topo_num}_info.txt')
    
    host_num = None
    switch_num = None
    connect_switch_order = None
    
    try:
        with open(info_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'host_num':
                    host_num = int(value)
                elif key == 'switch_num':
                    switch_num = int(value)
                elif key == 'connect_switch_order':
                    connect_switch_order = list(map(int, value.split(',')))
        
        if None in [host_num, switch_num, connect_switch_order]:
            raise ValueError("配置文件缺少必要信息")
        
        return host_num, switch_num, connect_switch_order
    
    except Exception as e:
        log_message(f"读取拓扑信息失败: {e}", 'ERROR')
        sys.exit(1)

# ==================== 接入交换机识别 ====================

def get_edge_switch_for_host(net, host_name):
    """
    动态识别指定主机的接入交换机
    """
    try:
        host = net.get(host_name)
        
        if not host:
            log_message(f"未找到主机: {host_name}", 'ERROR')
            return None
        
        for intf in host.intfList():
            if intf.link:
                peer_intf = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
                
                if hasattr(peer_intf, 'node') and hasattr(peer_intf.node, 'name'):
                    peer_name = peer_intf.node.name
                    
                    if peer_name.startswith('s'):
                        log_message(f"✓ 识别到 {host_name} 的接入交换机: {peer_name}")
                        return peer_name
        
        log_message(f"⚠ 未找到 {host_name} 的接入交换机", 'WARNING')
        return None
    
    except Exception as e:
        log_message(f"识别接入交换机失败: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        return None

def get_hosts_connected_to_switch(net, switch_name):
    """
    获取直接连接到指定交换机的所有主机
    
    参数:
        net: Mininet 网络对象
        switch_name: 交换机名称（例如 's0'）
    
    返回:
        connected_hosts: 主机名称列表（例如 ['h0', 'h1']）
    """
    switch = net.get(switch_name)
    
    if not switch:
        log_message(f"未找到交换机: {switch_name}", 'ERROR')
        return []
    
    connected_hosts = []
    
    for host in net.hosts:
        for intf in host.intfList():
            if intf.link:
                peer_intf = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
                
                if hasattr(peer_intf, 'node') and peer_intf.node == switch:
                    connected_hosts.append(host.name)
                    break
    
    log_message(f"  交换机 {switch_name} 连接的主机: {connected_hosts}")
    
    return connected_hosts

# ==================== 网络测试函数 ====================

def verify_pingall(net, max_attempts=2):
    """验证网络连通性"""
    for attempt in range(1, max_attempts + 1):
        loss = net.pingAll(timeout='2')
        if loss == 0:
            log_message(f"✓ 网络连通性验证通过（尝试 {attempt}/{max_attempts}）")
            return True
        log_message(f"⚠ 网络丢包率 {loss}%（尝试 {attempt}/{max_attempts}）", 'WARNING')
        time.sleep(2)
    
    log_message("✗ 网络连通性验证失败", 'ERROR')
    return False

def measure_single_pair(sender, receiver, duration=5):
    """
    测量单个主机对的吞吐量
    
    返回:
        throughput: 吞吐量（Mbits/sec）
    """
    port = random.randint(5000, 6000)
    
    # 清理旧进程
    sender.cmd('killall -9 iperf 2>/dev/null')
    receiver.cmd('killall -9 iperf 2>/dev/null')
    time.sleep(0.1)
    
    # 启动 iperf server
    receiver.cmd(f'iperf -s -p {port} > /dev/null 2>&1 &')
    time.sleep(0.3)
    
    throughput = 0.0
    
    try:
        cmd = f'timeout {duration+5} iperf -c {receiver.IP()} -p {port} -t {duration}'
        output = sender.cmd(cmd)
        
        for line in output.strip().split('\n'):
            if "Mbits/sec" in line and "sender" not in line.lower():
                try:
                    parts = line.split()
                    for i in range(len(parts)-1, -1, -1):
                        if "Mbits/sec" in parts[i]:
                            throughput = float(parts[i-1])
                            break
                except (ValueError, IndexError):
                    pass
    
    except Exception as e:
        pass
    
    finally:
        receiver.cmd('killall -9 iperf 2>/dev/null')
    
    return throughput

def measure_network_throughput_single_source(net, source_host='h0', duration=5):
    """
    单点发送测量：从指定主机发送到所有其他主机
    """
    source = net.get(source_host)
    total_throughput = 0.0
    successful_tests = 0
    failed_tests = 0
    
    for host in net.hosts:
        if host == source:
            continue
        
        throughput = measure_single_pair(source, host, duration)
        
        if throughput > 0:
            total_throughput += throughput
            successful_tests += 1
        else:
            failed_tests += 1
    
    return total_throughput, successful_tests, failed_tests

def measure_stratified_sampling(net, failed_switch=None, sample_ratio=0.3, duration=5):
    """
    分层采样测量（核心改进）
    
    参数:
        net: Mininet 网络对象
        failed_switch: 故障交换机名称（None 表示基准测试）
        sample_ratio: 采样比例
        duration: iperf 测试时长
    
    返回:
        estimated_total: 估算的总吞吐量（Mbits/sec）
        successful_pairs: 成功的测试对数
        total_sampled: 总采样对数
    
    逻辑:
        1. 识别受影响的主机（直连故障交换机）
        2. 将所有主机对分为两类：
           - A 类：涉及受影响主机的流
           - B 类：不涉及受影响主机的流
        3. 分别采样、测量、估算
        4. 汇总结果
    """
    hosts = net.hosts
    
    # 1. 识别受影响的主机
    if failed_switch:
        affected_host_names = get_hosts_connected_to_switch(net, failed_switch)
        log_message(f"  受影响主机: {affected_host_names}")
    else:
        affected_host_names = []  # 基准测试时无故障
    
    # 2. 分类所有主机对
    affected_pairs = []    # A 类：涉及受影响主机的流
    unaffected_pairs = []  # B 类：不涉及受影响主机的流
    
    for sender in hosts:
        for receiver in hosts:
            if sender == receiver:
                continue
            
            if sender.name in affected_host_names or receiver.name in affected_host_names:
                affected_pairs.append((sender, receiver))
            else:
                unaffected_pairs.append((sender, receiver))
    
    log_message(f"  流分类: A 类（受影响）{len(affected_pairs)} 对, "
               f"B 类（未受影响）{len(unaffected_pairs)} 对")
    
    # 3. 分别采样
    if len(affected_pairs) > 0:
        n_affected = max(int(len(affected_pairs) * sample_ratio), 1)
        sampled_affected = random.sample(affected_pairs, 
                                        min(n_affected, len(affected_pairs)))
    else:
        sampled_affected = []
    
    if len(unaffected_pairs) > 0:
        n_unaffected = max(int(len(unaffected_pairs) * sample_ratio), 1)
        sampled_unaffected = random.sample(unaffected_pairs, 
                                          min(n_unaffected, len(unaffected_pairs)))
    else:
        sampled_unaffected = []
    
    log_message(f"  采样数量: A 类 {len(sampled_affected)} 对, "
               f"B 类 {len(sampled_unaffected)} 对")
    
    # 4. 测量 A 类（受影响流）
    throughput_affected = 0.0
    successful_affected = 0
    
    for idx, (sender, receiver) in enumerate(sampled_affected, 1):
        throughput = measure_single_pair(sender, receiver, duration)
        if throughput > 0:
            throughput_affected += throughput
            successful_affected += 1
        
        if idx % 3 == 0 or idx == len(sampled_affected):
            sys.stdout.write(f'\r    测试 A 类: {idx}/{len(sampled_affected)} 对')
            sys.stdout.flush()
    
    if len(sampled_affected) > 0:
        print()
    
    # 5. 测量 B 类（未受影响流）
    throughput_unaffected = 0.0
    successful_unaffected = 0
    
    for idx, (sender, receiver) in enumerate(sampled_unaffected, 1):
        throughput = measure_single_pair(sender, receiver, duration)
        if throughput > 0:
            throughput_unaffected += throughput
            successful_unaffected += 1
        
        if idx % 3 == 0 or idx == len(sampled_unaffected):
            sys.stdout.write(f'\r    测试 B 类: {idx}/{len(sampled_unaffected)} 对')
            sys.stdout.flush()
    
    if len(sampled_unaffected) > 0:
        print()
    
    # 6. 分别估算
    if len(sampled_affected) > 0:
        estimated_affected = throughput_affected * (len(affected_pairs) / len(sampled_affected))
    else:
        estimated_affected = 0.0
    
    if len(sampled_unaffected) > 0:
        estimated_unaffected = throughput_unaffected * (len(unaffected_pairs) / len(sampled_unaffected))
    else:
        estimated_unaffected = 0.0
    
    # 7. 汇总
    total_estimated = estimated_affected + estimated_unaffected
    total_successful = successful_affected + successful_unaffected
    total_sampled = len(sampled_affected) + len(sampled_unaffected)
    
    log_message(f"  分层估算结果:")
    log_message(f"    A 类: 实测 {throughput_affected:.2f} Mbps "
               f"(成功 {successful_affected}/{len(sampled_affected)}) "
               f"→ 估算 {estimated_affected:.2f} Mbps")
    log_message(f"    B 类: 实测 {throughput_unaffected:.2f} Mbps "
               f"(成功 {successful_unaffected}/{len(sampled_unaffected)}) "
               f"→ 估算 {estimated_unaffected:.2f} Mbps")
    log_message(f"    总计: {total_estimated:.2f} Mbps")
    
    return total_estimated, total_successful, total_sampled

# ==================== 故障模拟 ====================

def simulate_switch_failure_v2(net, switch_name):
    """模拟交换机完全故障"""
    switch = net.get(switch_name)
    
    log_message(f"✗ 模拟 {switch_name} 完全故障")
    
    try:
        ports_output = switch.cmd(f'ovs-ofctl show {switch_name} -O OpenFlow13')
        ports = []
        for line in ports_output.split('\n'):
            if '(' in line and 'addr' in line:
                port_num = line.strip().split('(')[0].strip()
                if port_num.isdigit():
                    ports.append(port_num)
        
        log_message(f"  检测到 {len(ports)} 个端口")
        
        switch.cmd(f'ovs-ofctl del-flows {switch_name} -O OpenFlow13')
        log_message(f"  ✓ 已删除所有流表")
        
        controller_info = switch.cmd(f'ovs-vsctl get-controller {switch_name}').strip()
        switch.cmd(f'ovs-vsctl del-controller {switch_name}')
        log_message(f"  ✓ 已断开控制器连接")
        
        switch.cmd(f'ovs-ofctl add-flow {switch_name} "priority=0,actions=drop" -O OpenFlow13')
        log_message(f"  ✓ 已设置默认 DROP 规则")
        
        for port in ports:
            switch.cmd(f'ovs-ofctl mod-port {switch_name} {port} down -O OpenFlow13')
        log_message(f"  ✓ 已禁用所有端口")
        
        log_message(f"✗ {switch_name} 故障模拟完成")
        
        return {
            'switch': switch,
            'switch_name': switch_name,
            'ports': ports,
            'controller_ip': controller_info
        }
        
    except Exception as e:
        log_message(f"故障模拟失败: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        return None

def restore_switch_v2(net, failure_info):
    """恢复交换机"""
    if not failure_info:
        return
    
    switch = failure_info['switch']
    switch_name = failure_info['switch_name']
    
    log_message(f"✓ 恢复 {switch_name}...")
    
    try:
        for port in failure_info['ports']:
            switch.cmd(f'ovs-ofctl mod-port {switch_name} {port} up -O OpenFlow13')
        log_message(f"  ✓ 已启用 {len(failure_info['ports'])} 个端口")
        
        switch.cmd(f'ovs-ofctl del-flows {switch_name} -O OpenFlow13')
        log_message(f"  ✓ 已删除 DROP 规则")
        
        if failure_info['controller_ip']:
            switch.cmd(f'ovs-vsctl set-controller {switch_name} {failure_info["controller_ip"]}')
            log_message(f"  ✓ 已重新连接控制器")
        
        time.sleep(2)
        
        log_message(f"  等待流表重建...")
        trigger_controller_learning(net)
        
        time.sleep(3)
        
        log_message(f"✓ {switch_name} 已完全恢复")
        
    except Exception as e:
        log_message(f"恢复失败: {e}", 'ERROR')
        import traceback
        traceback.print_exc()

def trigger_controller_learning(net):
    """触发控制器重新学习"""
    log_message(f"  发送探测流量...")
    
    for host in net.hosts:
        host.cmd('ip -s -s neigh flush all > /dev/null 2>&1')
        host.cmd(f'arping -c 2 -U -I {host.defaultIntf()} {host.IP()} > /dev/null 2>&1 &')
        
        for other_host in net.hosts:
            if other_host != host:
                host.cmd(f'ping -c 1 -W 1 {other_host.IP()} > /dev/null 2>&1 &')
    
    time.sleep(2)

# ==================== 主实验流程 ====================

def run_brute_force_experiment_stratified(topo_num):
    """
    执行暴力故障影响分析实验（分层采样版本）
    """
    log_message(f"========== 开始实验：拓扑 {topo_num} ==========")
    
    # 1. 加载拓扑
    host_n, switch_n, edge_switch = get_topo_info(topo_num)
    topo_file = os.path.join(CONFIG['topo_base_path'], 
                            f'topo_{topo_num}_result/topo_{topo_num}.txt')
    
    topo = CustomTopo(host_n, switch_n, filePath=topo_file, edge_switch=edge_switch)
    
    # 2. 创建网络
    controller = RemoteController('c0', 
                                  ip=CONFIG['controller_ip'], 
                                  port=CONFIG['controller_port'])
    net = Mininet(
        topo=topo,
        controller=controller,
        host=CPULimitedHost,
        link=TCLink,
        switch=partial(OVSSwitch, protocols='OpenFlow13', stp=False),
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    # 3. 启动网络
    net.start()
    log_message("Mininet 网络已启动")
    
    # 4. 等待路由收敛
    log_message(f"等待路由收敛（{CONFIG['routing_convergence_time']} 秒）...")
    for i in range(CONFIG['routing_convergence_time']):
        progress_bar(i+1, CONFIG['routing_convergence_time'], prefix='路由收敛')
        time.sleep(1)
    
    # 5. 验证网络连通性
    if not verify_pingall(net, max_attempts=CONFIG['pingall_max_attempts']):
        log_message("网络初始化失败，终止实验", 'ERROR')
        net.stop()
        return None
    
    # 6. 动态识别 h0 的接入交换机
    log_message("========== 识别 h0 的接入交换机 ==========")
    edge_switch_h0 = get_edge_switch_for_host(net, 'h0')
    
    if not edge_switch_h0:
        log_message("无法识别 h0 的接入交换机，终止实验", 'ERROR')
        net.stop()
        return None
    
    log_message(f"⚠ {edge_switch_h0} 将使用分层采样多对多测量")
    
    # 7. 基准测试
    log_message("========== 阶段 1: 基准测试 ==========")
    
    # 7.1 单点发送基准
    log_message("基准测试 1: 单点发送（h0 → 所有主机）")
    baseline_single_throughput, baseline_single_success, baseline_single_fail = \
        measure_network_throughput_single_source(net, source_host='h0', 
                                                duration=CONFIG['iperf_duration'])
    log_message(f"✓ 单点发送基准: {baseline_single_throughput:.2f} Mbits/sec "
               f"(成功 {baseline_single_success}, 失败 {baseline_single_fail})")
    
    if baseline_single_throughput == 0:
        log_message("单点基准吞吐量为 0，终止实验", 'ERROR')
        net.stop()
        return None
    
    # 7.2 分层采样多对多基准
    log_message(f"基准测试 2: 分层采样多对多（仅用于 {edge_switch_h0}）")
    baseline_multi_throughput, baseline_multi_success, baseline_multi_total = \
        measure_stratified_sampling(
            net, 
            failed_switch=None,  # 基准测试无故障
            sample_ratio=CONFIG['multi_pair_sample_ratio'],
            duration=CONFIG['iperf_duration']
        )
    log_message(f"✓ 多对多基准: {baseline_multi_throughput:.2f} Mbits/sec "
               f"(成功 {baseline_multi_success}/{baseline_multi_total} 对)")
    
    if baseline_multi_throughput == 0:
        log_message("多对多基准吞吐量为 0，终止实验", 'ERROR')
        net.stop()
        return None
    
    # 8. 暴力遍历
    log_message("========== 阶段 2: 暴力故障遍历 ==========")
    results = []
    switches = net.switches
    total_switches = len(switches)
    
    for idx, switch in enumerate(switches, 1):
        switch_name = switch.name
        log_message(f"{'='*60}")
        log_message(f"测试 {switch_name} ({idx}/{total_switches})")
        log_message(f"{'='*60}")
        
        # 8.1 判断是否为 h0 的接入交换机
        is_h0_edge_switch = (switch_name == edge_switch_h0)
        
        if is_h0_edge_switch:
            log_message(f"  ⚠ {switch_name} 是 h0 的接入交换机，使用分层采样多对多测量")
        else:
            log_message(f"  使用单点测量（h0 → 所有主机）")
        
        # 8.2 模拟故障
        failure_info = simulate_switch_failure_v2(net, switch_name)
        
        if not failure_info:
            log_message(f"跳过 {switch_name}（故障模拟失败）", 'WARNING')
            progress_bar(idx, total_switches, prefix='故障遍历进度')
            continue
        
        # 8.3 等待收敛
        log_message(f"等待网络收敛（{CONFIG['convergence_wait']}秒）...")
        time.sleep(CONFIG['convergence_wait'])
        
        # 8.4 验证故障效果
        log_message("验证故障效果...")
        loss = net.pingAll(timeout='1')
        log_message(f"PingAll 丢包率: {loss}%")
        
        # 8.5 测量故障后吞吐量
        if is_h0_edge_switch:
            # h0 的接入交换机：使用分层采样多对多测量
            log_message(f"测量故障后吞吐量（分层采样多对多）...")
            fail_throughput, fail_success, fail_total = \
                measure_stratified_sampling(
                    net,
                    failed_switch=switch_name,  # 关键：传入故障交换机
                    sample_ratio=CONFIG['multi_pair_sample_ratio'],
                    duration=CONFIG['iperf_duration']
                )
            
            baseline_throughput = baseline_multi_throughput
            measurement_type = 'stratified_multi_pair'
            
        else:
            # 其他交换机：使用单点发送测量
            log_message(f"测量故障后吞吐量（单点发送）...")
            fail_throughput, fail_success, fail_total = \
                measure_network_throughput_single_source(
                    net,
                    source_host='h0',
                    duration=CONFIG['iperf_duration']
                )
            
            baseline_throughput = baseline_single_throughput
            measurement_type = 'single_source'
        
        # 8.6 计算影响分数
        impact_score = (baseline_throughput - fail_throughput) / baseline_throughput \
                       if baseline_throughput > 0 else 0
        
        log_message(f"✓ 影响分数: {impact_score:.4f} (测量类型: {measurement_type})")
        log_message(f"  基准: {baseline_throughput:.2f}, 故障: {fail_throughput:.2f}")
        
        # 8.7 恢复
        restore_switch_v2(net, failure_info)
        time.sleep(CONFIG['recovery_wait'])
        
        # 8.8 验证恢复
        log_message("验证网络恢复...")
        if not verify_pingall(net, max_attempts=2):
            log_message(f"⚠ 网络恢复异常", 'WARNING')
        
        # 8.9 记录结果
        results.append({
            'switch_id': switch_name,
            'is_h0_edge_switch': bool(is_h0_edge_switch),
            'measurement_type': measurement_type,
            'impact_score': float(impact_score),
            'packet_loss': float(loss) / 100.0,
            'throughput_baseline': float(baseline_throughput),
            'throughput_failure': float(fail_throughput),
            'throughput_loss': float(baseline_throughput - fail_throughput),
            'successful_tests': int(fail_success),
            'total_tests': int(fail_total),
        })
        
        progress_bar(idx, total_switches, prefix='故障遍历进度')
    
    # 9. 停止网络
    net.stop()
    log_message("Mininet 网络已停止")
    
    # 10. 保存原始结果
    raw_results_file = os.path.join(CONFIG['output_dir'], 
                                    f'raw_results_{topo_num}.json')
    with open(raw_results_file, 'w') as f:
        json.dump(results, f, indent=4)
    log_message(f"✓ 原始结果已保存: {raw_results_file}")
    
    return results

def integrate_with_topsis(topo_num, brute_force_results):
    """整合 TOPSIS 结果"""
    log_message("========== 阶段 3: 整合 TOPSIS 结果 ==========")
    
    metrics_file = os.path.join(CONFIG['topo_base_path'], 
                               f'topo_{topo_num}_result/metrics.txt')
    
    if not os.path.exists(metrics_file):
        log_message(f"指标文件不存在: {metrics_file}", 'ERROR')
        return None
    
    try:
        critical_nodes = identify_key_nodes_adaptive(metrics_file)
        log_message(f"✓ TOPSIS 识别完成，识别出 {len(critical_nodes)} 个关键节点")
    except Exception as e:
        log_message(f"TOPSIS 识别失败: {e}", 'ERROR')
        return None
    
    topsis_scores = {}
    topsis_ranks = {}
    for rank, (node_name, score) in enumerate(critical_nodes, 1):
        topsis_scores[node_name] = score
        topsis_ranks[node_name] = rank
    
    integrated_results = []
    for result in brute_force_results:
        switch_id = result['switch_id']
        integrated_results.append({
            'switch_id': switch_id,
            'topsis_score': float(topsis_scores.get(switch_id, 0.0)),
            'topsis_rank': int(topsis_ranks.get(switch_id, len(critical_nodes) + 1)),
            'is_h0_edge_switch': result['is_h0_edge_switch'],
            'measurement_type': result['measurement_type'],
            'impact_score': result['impact_score'],
            'packet_loss': result['packet_loss'],
            'throughput_baseline': result['throughput_baseline'],
            'throughput_failure': result['throughput_failure'],
            'throughput_loss': result['throughput_loss'],
            'successful_tests': result['successful_tests'],
            'total_tests': result['total_tests'],
        })
    
    integrated_results.sort(key=lambda x: x['topsis_rank'])
    
    csv_file = os.path.join(CONFIG['output_dir'], 
                           f'integrated_results_{topo_num}.csv')
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = list(integrated_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(integrated_results)
    
    log_message(f"✓ 整合结果已保存: {csv_file}")
    
    # 打印 Top-10
    print("\n" + "="*120)
    print("Top-10 关键节点对比（TOPSIS 排名）")
    print("="*120)
    print(f"{'排名':<6} {'节点':<10} {'TOPSIS':<10} {'实际影响':<12} "
          f"{'丢包率':<10} {'测量类型':<25} {'h0接入':<8}")
    print("-"*120)
    for i, result in enumerate(integrated_results[:10], 1):
        h0_edge_mark = '✓' if result['is_h0_edge_switch'] else ''
        print(f"{i:<6} {result['switch_id']:<10} "
              f"{result['topsis_score']:<10.4f} "
              f"{result['impact_score']:<12.4f} "
              f"{result['packet_loss']*100:<10.1f}% "
              f"{result['measurement_type']:<25} "
              f"{h0_edge_mark:<8}")
    print("="*120)
    print("注：✓ 表示 h0 的接入交换机（使用分层采样）\n")
    
    return integrated_results

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║   暴力故障影响分析实验 - 分层采样版本                      ║
║                                                          ║
║  核心改进:                                               ║
║    1. ✓ 动态识别 h0 的接入交换机                          ║
║    2. ✓ 分层采样避免估算偏差                              ║
║    3. ✓ 数学上无偏的估计量                                ║
║    4. ✓ 时间成本不增加                                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n⚠️  请确保 Ryu 控制器已启动:")
    print(f"   $ ryu-manager my_learning_switch.py")
    print(f"   控制器地址: {CONFIG['controller_ip']}:{CONFIG['controller_port']}\n")
    
    input("按 Enter 继续...")
    
    topo_num = input("\n请输入拓扑编号（例如 1）: ").strip()
    
    topo_file = os.path.join(CONFIG['topo_base_path'], 
                            f'topo_{topo_num}_result/topo_{topo_num}.txt')
    if not os.path.exists(topo_file):
        log_message(f"拓扑文件不存在: {topo_file}", 'ERROR')
        return
    
    start_time = time.time()
    
    try:
        brute_force_results = run_brute_force_experiment_stratified(topo_num)
        
        if not brute_force_results:
            log_message("暴力实验失败", 'ERROR')
            return
        
        integrated_results = integrate_with_topsis(topo_num, brute_force_results)
        
        if not integrated_results:
            log_message("结果整合失败", 'ERROR')
            return
        
        elapsed_time = time.time() - start_time
        log_message(f"========== 实验完成 ==========")
        log_message(f"总耗时: {elapsed_time/60:.2f} 分钟")
        log_message(f"结果文件保存在: {CONFIG['output_dir']}")
        
        print(f"\n✓ 实验完成！\n")
    
    except KeyboardInterrupt:
        log_message("\n实验被用户中断", 'WARNING')
        cleanup_mininet()
    
    except Exception as e:
        log_message(f"实验发生异常: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        cleanup_mininet()

if __name__ == '__main__':
    main()
