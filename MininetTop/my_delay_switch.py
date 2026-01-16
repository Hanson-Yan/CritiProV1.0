from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib.packet import ether_types
from ryu.lib import stplib
import threading
import time
import random
import os

# 获取 critipro 延迟
topo_num="topo_2"
file=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/deploy_cost/10000.txt"
def get_critipro_delay(filepath=file):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':')
                val = val.strip(' ()\n')
                parts = val.split(',')
                result[key.strip()] = float(parts[1])
    return result.get('critipro', 0)

# print(get_critipro_delay())


class MyL2Switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'stplib': stplib.Stp}

    def __init__(self, *args, **kwargs):
        super(MyL2Switch, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.stp = kwargs['stplib']
        self.delay = get_critipro_delay()  # 一次性读取延迟值
        self.delay_enabled = False
        # self.start_time = time.time()
        # self.activation_delay = 70  # 延迟激活延迟逻辑，比如30秒
        # self.activation_delay = 0  # 延迟激活延迟逻辑，比如30秒
        self.signal_file = "/tmp/enable_delay_signal"  # IPC 文件路径

    def check_activation_signal(self):
        if not self.delay_enabled and os.path.exists(self.signal_file):
            self.logger.info("📡 Received delay activation signal from Mininet.")
            self.delay_enabled = True


    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):

        self.check_activation_signal()
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        if src in self.mac_to_port[dpid]:
            if self.mac_to_port[dpid][src] != in_port:
                self.logger.warning(f"[LOOP WARNING] {src} changed port.")
                return
        else:
            self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # 延迟逻辑：s0（DPID 1）中 1% 的数据包
        # if not self.delay_enabled and time.time() - self.start_time > self.activation_delay:
        #     self.delay_enabled = True

        if self.delay_enabled and dpid == 1 and random.random() < 0.1:
            # self.logger.info(f"Delaying packet from s0 by {self.delay} seconds")
            # time.sleep(self.delay)
            self.logger.info(f"Scheduling delay of {self.delay} seconds for s0 packet")
            threading.Thread(target=time.sleep, args=(self.delay,), daemon=True).start()

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
