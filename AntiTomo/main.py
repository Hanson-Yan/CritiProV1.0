from generate_matrix import AntiTomoDefender
from matrix_to_vector import DelayMatrixProcessor
from adjacency_to_routing import run_ad_to_rout,get_user_input
import numpy as np
from confuse_matrix import ProTO
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

def compare_and_adjust(A, B):
    """
    比较两个矩阵 A 和 B：
    - 如果 B 的列数小于 A 的列数，对 B 进行零填充，使其列数与 A 相同。
    - 如果 B 的列数大于 A 的列数，从列下标最大的开始删除，直到两矩阵列数相同。
    """
    A_cols = A.shape[1]  # 矩阵 A 的列数
    B_cols = B.shape[1]  # 矩阵 B 的列数

    if B_cols < A_cols:
        # 如果 B 的列数小于 A 的列数，对 B 进行零填充
        B = np.pad(B, ((0, 0), (0, A_cols - B_cols)), mode='constant')
    elif B_cols > A_cols:
        # 如果 B 的列数大于 A 的列数，从列下标最大的开始删除
        cols_to_delete = range(B_cols - 1, A_cols - 1, -1)  # 从最大的列下标开始删除
        B = np.delete(B, cols_to_delete, axis=1)  # 删除指定的列

    return B

# def load_data():
#     print("---------------加载数据---------------")
#     input_file_dir="/home/retr0/Project/TopologyObfu/AntiTomo/input_file/"
#     critipro_path="/home/retr0/Project/TopologyObfu/AntiTomo/"
#     topo_num=input("please input topo_num:")
#     suffix_result="_result/"
#     suffix_txt=".txt"
#     suffix_info="_info.txt"
#     delay_matirx_txt="delay_matrix.txt"
#     topo_matrix_original_txt="topo_matrix_original.txt"
#     input_info_txt="topo_info.txt"

#     topo_num_result_dir=critipro_path+topo_num+suffix_result
#     topo_num_topo=topo_num_result_dir+topo_num+suffix_txt
#     topo_num_info=topo_num_result_dir+topo_num+suffix_info
#     topo_num__delay=topo_num_result_dir+delay_matirx_txt

#     if not os.path.exists(topo_num_result_dir):
#         print(f"{topo_num_result_dir} is not exist")
#         sys.exit(1)
#     if not os.path.exists(topo_num_topo):
#         print(f"{topo_num_topo} is not exist")
#         sys.exit(1)
#     if not os.path.exists(topo_num__delay):
#         print(f"{topo_num__delay} is not exist")
#         sys.exit(1)
#     if not os.path.exists(topo_num_info):
#         print(f"{topo_num_info} is not exist")
#         sys.exit(1)

    
#     input_file_topo=input_file_dir+topo_matrix_original_txt
#     input_file_delay=input_file_dir+delay_matirx_txt
#     input_file_info=input_file_dir+input_info_txt
    
#     copy_file(topo_num_topo,input_file_topo)
#     copy_file(topo_num_info,input_file_info)
#     copy_file(topo_num__delay,input_file_delay)

def prepare_data(root_node,receiver_node,topo_num,prob_num):

    original_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/{topo_num}.txt"
    original_routing_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/routing_matrix.txt"
    delay_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/delay_result/{prob_num}/{topo_num}_simu_delay.txt"
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/delay_vector.txt"
    print("原始拓扑邻接矩阵转路由矩阵\n")
    run_ad_to_rout(original_adj_matrix_path,original_routing_matrix_path,root_node,receiver_node)

    print("---------------延迟矩阵提取相关延迟向量---------------")
    delay_vector_extractor=DelayMatrixProcessor(delay_matrix_path,delay_vector_path)
    delay_vector_extractor.extract_non_zero_delays()


def antitomo_exe(root_node,receiver_node,topo_num):
    original_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/{topo_num}.txt"
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/delay_vector.txt"
    print("---------------开始拓扑混淆---------------")
    original_adj_matrix = np.loadtxt(original_adj_matrix_path)
    delay_vector=np.loadtxt(delay_vector_path)
    # 创建AntiTomo实例
    params = {
        'phi_n': 3,
        # 'phi_n': 8,
        'phi_l': 6,
        # 'phi_l': 16,
        'v_min': 0.1,
        'v_max': 3.0,
        'delta_max': 200,
        'lambda_simi': 0.5,
        'lambda_cost': 0.5,
        'sigma': 5,
        'w_candidates': 1000
    }
    
   
    # root_node,receiver_node=get_user_input()
    # 初始化防御系统
    defender = AntiTomoDefender(original_adj_matrix, delay_vector, receiver_node, params)
    # 生成混淆拓扑
    best_candidate,best_core,best_delays = defender.generate_obfuscated_topology()
    
    # 输出结果
    if best_candidate:
        print("最优混淆拓扑邻接矩阵:\n", best_candidate['adj_matrix'])
        print("目标函数值:", best_core)
        print("best_delays:", best_delays)
    else:
        print("未找到有效混淆拓扑")
    confuse_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/topo_matrix_confuse.txt"
    confuse_routing_matrix_path = f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/confuse_routing_matrix.txt"
    np.savetxt(confuse_adj_matrix_path,best_candidate['adj_matrix'],fmt="%d")
    print(f"混淆拓扑邻接矩阵已保存至\n{confuse_adj_matrix_path}")
    print("混淆拓扑邻接矩阵转路由矩阵\n")

    run_ad_to_rout(confuse_adj_matrix_path,confuse_routing_matrix_path,root_node,receiver_node)


def deploy_solve(topo_num,prob_num):
    original_routing_matrix_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/routing_matrix.txt"
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/delay_vector.txt"
    confuse_routing_matrix_path = f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/confuse_routing_matrix.txt"
    deploy_vector_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
    print("---------------开始部署求解---------------")
    original_routing_matrix = np.loadtxt(original_routing_matrix_path)
    delay_vector=np.loadtxt(delay_vector_path)
    # print(original_routing_matrix.shape)
    max_delay_deviation = 50
    
    # return
    # 创建ProTO实例
    proto = ProTO(original_routing_matrix, delay_vector,max_delay_deviation)
    # 生成虚假拓扑矩阵 A_m
    confuse_routing_matrix= np.loadtxt(confuse_routing_matrix_path)
    confuse_routing_matrix = compare_and_adjust(original_routing_matrix,confuse_routing_matrix)
    print("虚假拓扑矩阵 A_m:")
    print(confuse_routing_matrix)
    
    # 计算操纵矩阵 F
    print(f"confuse_routing_matrix.shape:{confuse_routing_matrix.shape}")
    # return
    F = proto.solve_optimization(confuse_routing_matrix)

    print("操纵矩阵 F:")
    print(F)
    # print(F.shape)

    deploy_vector = proto.compute_Fx(F)
    print("部署延迟向量 Fx:")
    print(deploy_vector)
    np.savetxt(deploy_vector_path,deploy_vector,fmt="%.6f")
    print(f"部署延迟向量已保存至\n{deploy_vector_path}")

def prepare_delay_matrix(topo_num,prob_num_arr):
    delay_result = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/"
    topo_num_result = f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/delay_result/"
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

# 示例用法
if __name__ == "__main__":
    prob_num_arr = [500,1000,2000,3000,5000,7000,10000]
    topo_num=input("please input topo_num:")
    info_path=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/{topo_num}_info.txt"
    prepare_delay_matrix(topo_num=topo_num,prob_num_arr=prob_num_arr)
    for prob_num in prob_num_arr:
    #     load_data()
        print("---------------邻接矩阵转路由矩阵---------------")
        ost_num, switch_num, connect_switch_order=get_user_input(info_path)
        root_node=connect_switch_order[0]
        receiver_node=connect_switch_order[1:]
        prepare_data(root_node,receiver_node,topo_num,prob_num)
        antitomo_exe(root_node,receiver_node,topo_num)
        deploy_solve(topo_num,prob_num)
        # break
