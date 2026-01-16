h1_results中，前10个为（h1,h2) 接受对的小包结果，后续10个为(h1,h3)接受对的小包结果，以此类推
h2_results中，前10个为（h2,h3) 接受对的小包结果，后续10个为(h2,h4)接受对的小包结果，以此类推
运行rnj来推拓扑：sudo /home/retr0/anaconda3/bin/python topo_infer_rnj.py


连续两个包同时发向同一目的地时都会存在40ms左右的误差，考虑是仿真的问题
发送端的一条sendp语句会引发额外的40ms左右延迟误差
现在夹层探测数据包直接一条sendp直接发，避免误差
且每组探测的接收点，单独运行发送程序，如1,2\1,3\1,4\1,5单独使用发送程序，2,3\2,4\2,5将重启发送程序
发送程序的格式为:python3 tps.py all_receives_num recevie_num probe_num
接收程序的格式为:python3 tpr.py h1-eth0 probe_num

探测实现需要xterm手动实现，避免误差

