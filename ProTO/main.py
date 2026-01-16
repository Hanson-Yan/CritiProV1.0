from confuse_matrix import ProTO
from matrix_to_vector import DelayMatrixProcessor
from adjacency_to_routing import run_ad_to_rout,get_user_input
import numpy as np
from generate_matrix import ObfuscatedTopologyGenerator
from draw_topo import DrawTopology

import os
import sys



def copy_file(source_file, target_file):
        try:
            # 打开源文件用于读取
            with open(source_file, 'r') as src:
                content = src.read()  # 读取源文件内容

            # 打开目标文件用于写入（覆盖）
            with open(target_file, 'w') as tgt:
                tgt.write(content)  # 将源文件内容写入目标文件

            print(f"成功将 {source_file} 的内容拷贝到 {target_file}")
        except FileNotFoundError:
            print(f"错误：文件 {source_file} 未找到")
        except Exception as e:
            print(f"发生错误：{e}")

def load_data(topo_num,prob_num):
    print("---------------加载数据---------------")
    
    input_file_dir="/home/retr0/Project/TopologyObfu/ProTO/input_file"
    topo_num_result_path=f"/home/retr0/Project/TopologyObfu/ProTO/"
    suffix_result="_result/"
    suffix_txt=".txt"
    suffix_info="_info.txt"
    delay_matirx_txt=f"delay_matrix.txt"
    topo_matrix_original_txt="topo_matrix_original.txt"
    topo_info_txt="topo_info.txt"

    topo_num_result_dir=topo_num_result_path+topo_num+suffix_result
    topo_num_topo=topo_num_result_dir+topo_num+suffix_txt
    topo_num_info=topo_num_result_dir+topo_num+suffix_info
    topo_num_delay=topo_num_result_dir+"delay_result/"+f"{prob_num}/"+f"{topo_num}_simu_delay.txt"

    if not os.path.exists(topo_num_result_dir):
        print(f"{topo_num_result_dir} is not exist")
        sys.exit(1)
    if not os.path.exists(topo_num_topo):
        print(f"{topo_num_topo} is not exist")
        sys.exit(1)
    if not os.path.exists(topo_num_delay):
        print(f"{topo_num_delay} is not exist")
        sys.exit(1)
    if not os.path.exists(topo_num_info):
        print(f"{topo_num_info} is not exist")
        sys.exit(1)

    
    input_file_topo=input_file_dir+topo_matrix_original_txt
    input_file_info=input_file_dir+topo_info_txt
    input_file_delay=input_file_dir+delay_matirx_txt
    
    copy_file(topo_num_topo,input_file_topo)
    copy_file(topo_num_info,input_file_info)
    copy_file(topo_num_delay,input_file_delay)

def prepare_delay_matrix(topo_num,prob_num_arr):
    delay_result = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/"
    topo_num_result = f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/delay_result/"
    for prob_num in prob_num_arr:
        topo_num_result_prob_num = f"{topo_num_result}{prob_num}/"
        try:
            os.makedirs(topo_num_result_prob_num, exist_ok=True)  # 如果文件夹已存在，不会报错
            print(f"文件夹 {topo_num_result_prob_num} 已创建成功！")
        except Exception as e:
            print(f"创建文件夹时发生错误：{e}")
        delay_prob_num=f"{delay_result}{prob_num}/{topo_num}_simu_delay.txt"
        topo_num_result_delay=f"{topo_num_result_prob_num}{topo_num}_simu_delay.txt"
        copy_file(delay_prob_num,topo_num_result_delay)

def prepare_data(topo_num,prob_num):

    print("---------------随机生成混淆拓扑邻接矩阵---------------")
    original_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/{topo_num}.txt"
    original_adj_matrix = np.loadtxt(original_adj_matrix_path)
    confuse_adj_matrix = ObfuscatedTopologyGenerator(original_adj_matrix).generate_obfuscated_adj_matrix()
    confuse_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/topo_matrix_confuse.txt"
    np.savetxt(confuse_adj_matrix_path,confuse_adj_matrix,fmt="%d")
    print(f"混淆拓扑邻接矩阵已经保存至\n{confuse_adj_matrix_path}")
    # critical_node=['s0','s1']
    # DrawTopology(confuse_adj_matrix,critical_node).draw()

    print("---------------邻接矩阵转路由矩阵---------------")
    # root_node,receiver_node=get_user_input()
    info_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/{topo_num}_info.txt"
    ost_num, switch_num, connect_switch_order=get_user_input(info_path)
    root_node=connect_switch_order[0]
    receiver_node=connect_switch_order[1:]
    print("原始拓扑邻接矩阵转路由矩阵\n")
    original_routing_matrix_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/routing_matrix.txt"
    run_ad_to_rout(original_adj_matrix_path,original_routing_matrix_path,root_node,receiver_node)
    print("混淆拓扑邻接矩阵转路由矩阵\n")
    confuse_routing_matrix_path =f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/confuse_routing_matrix.txt"
    run_ad_to_rout(confuse_adj_matrix_path,confuse_routing_matrix_path,root_node,receiver_node)

    print("---------------延迟矩阵提取相关延迟向量---------------")
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/delay_vector.txt"
    delay_matrix_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/delay_result/{prob_num}/{topo_num}_simu_delay.txt"
    delay_vector_extractor=DelayMatrixProcessor(delay_matrix_path,delay_vector_path)
    delay_vector_extractor.extract_non_zero_delays()


def run_test(topo_num,prob_num):
    print("---------------开始拓扑混淆---------------")
    original_routing_matrix_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/routing_matrix.txt"
    confuse_routing_matrix_path =f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/confuse_routing_matrix.txt"
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/delay_vector.txt"
    original_routing_matrix = np.loadtxt(original_routing_matrix_path)
    delay_vector=np.loadtxt(delay_vector_path)
    # print(original_routing_matrix.shape)
    max_delay_deviation = 50
    # 创建ProTO实例
    proto = ProTO(original_routing_matrix, delay_vector,max_delay_deviation)
    # 生成虚假拓扑矩阵 A_m
    confuse_routing_matrix= np.loadtxt(confuse_routing_matrix_path)
    print("虚假拓扑矩阵 A_m:")
    print(confuse_routing_matrix)
    # 计算操纵矩阵 F
    print(f"confuse_routing_matrix.shape:{confuse_routing_matrix.shape}")
    F = proto.solve_optimization(confuse_routing_matrix)

    print("操纵矩阵 F:")
    print(F)
    # print(F.shape)

    deploy_vector = proto.compute_Fx(F)
    print("部署延迟向量 Fx:")
    print(deploy_vector)
    deploy_vector_path=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
    np.savetxt(deploy_vector_path,deploy_vector,fmt="%.6f")
    print(f"部署延迟向量已保存至\n{deploy_vector_path}")


# 示例用法
if __name__ == "__main__":
    prob_num_arr = [500,1000,2000,3000,5000,7000,10000]
    topo_num=input("please input topo_num:")
    prepare_delay_matrix(topo_num=topo_num,prob_num_arr=prob_num_arr)
    for prob_num in prob_num_arr:
        # load_data(topo_num,prob_num)
        prepare_data(topo_num,prob_num)
        run_test(topo_num,prob_num)
        # break