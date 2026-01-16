import time
import subprocess
import re

def parse_iperf_output(output):
    """
    解析 iperf 输出，提取Server Report后的统计结果
    支持 UDP 模式 (带宽 + 丢包率)
    """
    lines = output.strip().split("\n")
    bw, loss = None, None
    
    # 查找包含Server Report的行
    server_report_line = None
    for i, line in enumerate(lines):
        if "Server Report:" in line:
            # 记录Server Report所在行的索引
            server_report_line = i
            break
    
    # 如果找到Server Report，检查它后面的行
    if server_report_line is not None and server_report_line + 1 < len(lines):
        # Server Report的下一行包含我们需要的数据
        target_line = lines[server_report_line + 1]
        print(target_line)
        try:
            parts = target_line.split()
            
            # 提取吞吐量 (Mbits/sec)
            bw_index = parts.index("Mbits/sec")
            bw = float(parts[bw_index - 1])
            
            # 提取丢包率 (%)
            if "%" in target_line:
                # 找到包含%的部分
                loss_part = [p for p in parts if "%" in p][0]
                loss_str = loss_part.strip("()%")
                loss = float(loss_str)
            else:
                loss = 0.0
                
        except (ValueError, IndexError, Exception):
            # 如果解析失败，保持bw和loss为None
            pass
    
    return bw, loss


def parse_ping_output(output):
    """
    解析 ping 输出，提取平均 RTT
    """
    for line in output.split("\n"):
        if "rtt min/avg/max/mdev" in line:
            avg_rtt = float(line.split("/")[4])
            return avg_rtt
    return None

# def run_test(net, src, dst, attack=None, duration=10, udp=True):
#     """
#     并发运行业务流和攻击流，输出指标
#     :param net: Mininet 对象
#     :param src: 业务流源主机
#     :param dst: 业务流目的主机
#     :param attack: (attacker_src, attacker_dst) 或 None
#     :param duration: 测试时长
#     :param udp: 是否使用 UDP 模式
#     """
#     src_host = net.get(src)
#     dst_host = net.get(dst)

#     # 清理旧进程并启动业务流 server
#     dst_host.cmd("pkill -9 iperf")
#     proto_flag = "-u" if udp else ""
#     dst_host.cmd(f"iperf -s {proto_flag} -p 5001 &")

#     # 启动攻击流 server
#     if attack:
#         attacker_src, attacker_dst = attack
#         att_src = net.get(attacker_src)
#         att_dst = net.get(attacker_dst)
#         att_dst.cmd("pkill -9 iperf")
#         att_dst.cmd(f"iperf -s -u -p 5002 &")

#     time.sleep(1)

#     # 启动业务流 client
#     proto_flag = "-u" if udp else ""
#     iperf_cmd = f"iperf -c {dst_host.IP()} {proto_flag} -b 10M -t {duration} -p 5001"
#     business_proc = src_host.popen(
#         iperf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#     )

#     # 启动攻击流 client
#     attack_proc = None
#     if attack:
#         iperf_attack = f"iperf -c {att_dst.IP()} -u -b 500M -t {duration} -p 5002"
#         attack_proc = att_src.popen(
#             iperf_attack, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )

#     # 等待业务流完成并解析结果
#     out, _ = business_proc.communicate()
#     bw, loss = parse_iperf_output(out)

#     # 等攻击流完成
#     if attack_proc:
#         attack_proc.communicate()

#     # RTT (ping)
#     ping_output = src_host.cmd(f"ping -c 5 {dst_host.IP()}")
#     rtt = parse_ping_output(ping_output)

#     result = {
#         "throughput_Mbps": bw,
#         "loss_percent": loss,
#         "avg_RTT_ms": rtt,
#     }

#     return result
def run_test(net, src, dst, attack=None, duration=10, udp=True, attack_first=True, attack_warm=2, attack_extra=5, debug=False):
    """
    :param net: Mininet net
    :param src: str e.g. "h1"  (业务流源)
    :param dst: str e.g. "h3"  (业务流目的)
    :param attack: tuple (attacker_src, attacker_dst) 或 None
    :param duration: 业务流持续时间 (秒)
    :param udp: 业务流是否使用 UDP (True/False)
    :param attack_first: 如果 True, 攻击流先启动并持续 duration+attack_extra 秒
    :param attack_warm: 攻击流启动后等待多少秒再启动业务流 (only if attack_first True)
    :param attack_extra: 攻击流比业务流多跑多少秒
    :param debug: 打印 iperf 原始输出
    :return: dict {"throughput_Mbps":..., "loss_percent":..., "avg_RTT_ms":...}
    """
    src_host = net.get(src)
    dst_host = net.get(dst)

    # 清理旧进程并启动业务 server (TCP/UDP)
    dst_host.cmd("pkill -9 iperf")
    proto_flag = "-u" if udp else ""
    dst_host.cmd(f"iperf -s {proto_flag} -p 5001 &")

    att_src = att_dst = None
    attack_proc = None
    if attack:
        attacker_src, attacker_dst = attack
        att_src = net.get(attacker_src)
        att_dst = net.get(attacker_dst)
        # att_dst.cmd("pkill -9 iperf")
        # 攻击服务端用 UDP server (port 5002)
        att_dst.cmd("iperf -s -u -p 5002 &")

    time.sleep(1)  # ensure servers up

    # 如果需要攻击先发
    if attack and attack_first:
        # 启动攻击 client，持续 duration + attack_extra
        iperf_attack_cmd = f"iperf -c {att_dst.IP()} -u -b 500M -t {duration + attack_extra} -p 5002"
        attack_proc = att_src.popen(iperf_attack_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 等待攻击占满链路
        time.sleep(attack_warm)

    # 启动业务流 client (TCP or UDP)
    if udp:
        # 为了能看到竞争，在 UDP 场景下业务可能也要发大一些（或者切换到 TCP）
        iperf_cmd = f"iperf -c {dst_host.IP()} -u -b 10M -t {duration} -p 5001"
    else:
        iperf_cmd = f"iperf -c {dst_host.IP()} -t {duration} -p 5001"
    business_proc = src_host.popen(iperf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 如果不要求攻击先发，则此处并行启动攻击（并发开）
    if attack and not attack_first:
        iperf_attack_cmd = f"iperf -c {att_dst.IP()} -u -b 500M -t {duration} -p 5002"
        attack_proc = att_src.popen(iperf_attack_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 等待业务流完成并获取输出（client stdout）
    out, err = business_proc.communicate(timeout=duration + 10)
    # out = out + "\n" + err
    if debug:
        print("\n================= DEBUG iperf business (client stdout) =================")
        print(out)
        print("====================================================================\n")

    # 等待攻击流结束（如果我们启动了它且它还没结束）
    if attack_proc:
        try:
            attack_out, attack_err = attack_proc.communicate(timeout=duration + attack_extra + 10)
            if debug:
                print("\n================= DEBUG iperf attack (client stdout) =================")
                print(attack_out)
                print("====================================================================\n")
        except subprocess.TimeoutExpired:
            # kill if it blocks
            attack_proc.kill()
            if debug:
                print("[WARN] attack_proc timeout and killed")
    if att_dst:
        att_dst.cmd("pkill -9 iperf")
    dst_host.cmd("pkill -9 iperf")
    # 解析 iperf 输出（优先 Server Report）
    bw, loss = parse_iperf_output(out)

    # RTT (ping)
    ping_output = src_host.cmd(f"ping -c 5 {dst_host.IP()}")
    rtt = None
    # parse ping avg
    for line in ping_output.splitlines():
        if "rtt min/avg/max/mdev" in line:
            parts = line.split("=")[1].split("/")
            rtt = float(parts[1])
    if loss == None:
        loss = -1
    result = {
        "throughput_Mbps": bw,
        "loss_percent": loss/100,
        "avg_RTT_ms": rtt
    }
    return result

