先画图再探测。

开启双终端，一个激活ryu-env环境运行ryu控制器，一个使用base运行mininet。确保pingall全通的情况下再开始探测。等待一段时间，确保pingall全通的情况下再开始探测。

ryu:
ryu-manager --observe-links my_learning_switch.py

mininet：
清除历史主机缓存：sudo mn -c
运行：sudo /home/retr0/anaconda3/bin/python small_topo_test.py
打开主机命令提示界面：xterm h1

手动xterm逐一探测，无背景流量下，探测次数100起步。

生成拓扑，转成拓扑树：
python /home/retr0/Project/TopologyObfu/MininetTop/get_topo/topo_generator.py && python /home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_to_tree.py