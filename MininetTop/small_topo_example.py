from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from functools import partial
from mininet.cli import CLI
from mininet.log import setLogLevel

class MinimalTestTopo(Topo):
    def build(self):
        # 添加交换机和主机
        s0 = self.addSwitch('s0')
        s1 = self.addSwitch('s1')

        h0 = self.addHost('h0', ip='10.0.0.1/24')
        h1 = self.addHost('h1', ip='10.0.0.2/24')

        # 连接主机和交换机
        self.addLink(h0, s0)
        self.addLink(h1, s1)

        # 连接两个交换机
        self.addLink(s0, s1)

def run():
    topo = MinimalTestTopo()
    controller = RemoteController('c0', ip='127.0.0.1', port=6633)
    net = Mininet(
        topo=topo,
        controller=controller,
        switch=partial(OVSSwitch, protocols='OpenFlow13'),
        link=TCLink
    )

    net.start()
    print("*** 网络启动完成，尝试使用 h0 ping h1 测试连通性 ***")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
