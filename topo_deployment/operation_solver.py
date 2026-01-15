import numpy as np
# from operation_matrix import OperationMatirxSolver
from pgd_operation_matrix import PGDOperationMatrixSolver
from torch_operation_matrix import TorchOperationMatrixSolver
from adjacency_to_routing import run_ad_to_rout,get_user_input
from matrix_to_vector import DelayMatrixProcessor
from adam_operation_matrix import OperationMatrixAdamSolver
from vector_to_matrix import vector_to_upper_triangle_matrix
import os
import shutil
import matplotlib.pyplot as plt

# def generate_test_data(n=5, seed=42):
#     """
#     生成随机测试数据并保存到 TXT 文件
#     :param n: 矩阵大小
#     :param seed: 随机种子（保证可复现性）
#     """
#     np.random.seed(seed)

#     # 生成 0-1 路由矩阵 & 混淆矩阵
#     M = np.random.randint(0, 2, size=(n, n))
#     F = np.random.randint(0, 2, size=(n, n))
#     r = np.random.rand(n, 1)  # 随机延迟向量

#     # 保存到 TXT 文件
#     np.savetxt("M.txt", M, fmt='%d')
#     np.savetxt("F.txt", F, fmt='%d')
#     np.savetxt("r.txt", r, fmt='%.6f')

#     print("已生成测试数据: M.txt, F.txt, r.txt")
#     return M, F, r




def run_test(topo_num,prob_num,M_path,F_path,r_path,method, node_num):
    """
    读取数据并运行优化
    :param method: 选择优化方法 ("proximal" 或 "scipy")
    """
    print(f"\n==== 使用 {method} 方法求解 P ====")
    if method == "adam":
        solver = OperationMatrixAdamSolver(topo_num=topo_num,prob_num=prob_num,gamma=0.01, eta=0.1, alpha=0.03, max_iter=5000, tol=1e-4, delta_max=50)
        # 使用 Adam 方法进行优化
        P_adam, r_adam = solver.solve(M_path, F_path, r_path, method)
        print(P_adam)
        print(r_adam)
    elif method == "pgd":
        
        solver = PGDOperationMatrixSolver(topo_num=topo_num,prob_num=prob_num,gamma=0.01, alpha=0.03, delta_max=100)
        P_pgd,r_pgd= solver.solve(M_path, F_path, r_path)
        print(P_pgd)
        print(r_pgd)
    elif method == "torch":
        if node_num<10:
            max_iter = 5000
        elif node_num<30:
            max_iter = 7000
        elif node_num<50:
            max_iter = 10000
        else:
            max_iter = 20000
        solver =TorchOperationMatrixSolver(topo_num=topo_num,prob_num=prob_num, max_iter=max_iter)
        P_pgd,r_pgd= solver.solve(M_path, F_path, r_path)
        print(P_pgd)
        print(r_pgd)
    else:
        pass
        # solver = OperationMatirxSolver()
        # P_result,r_result = solver.solve(M_path, F_path, r_path, method=method)
        # print(P_result)
        # print(r_result)

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

def prepare_delay_matrix(topo_num,prob_num_arr):
    delay_result = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/"
    topo_num_result = f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/delay_result/"
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
        

def prepare_data(topo_num,prob_num,M_path,F_path,r_path):
    obfusion_topo_data=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/data/{topo_num}_output_file/"
    confuse_adj_matrix_path=f"{obfusion_topo_data}output_adj.txt"

    output_file=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_output_file/"
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    confuse_routing_matrix_path = f"{output_file}confuse_routing_matrix.txt"
    original_routing_matrix_path = f"{output_file}routing_matrix.txt"

    original_adj_matrix_path=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/{topo_num}.txt"
    original_topo_info_path=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/{topo_num}_info.txt"


    delay_matrix_path=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/delay_result/{prob_num}/{topo_num}_simu_delay.txt"
    delay_vector_path=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/delay_result/{prob_num}/{topo_num}_delay_vector.txt"
    print("---------------邻接矩阵转路由矩阵---------------")
    host_num, switch_num, connect_switch_order=get_user_input(original_topo_info_path)
    root_node=connect_switch_order[0]
    receiver_node=connect_switch_order[1:]
    print("原始拓扑邻接矩阵转路由矩阵\n")
    run_ad_to_rout(original_adj_matrix_path,original_routing_matrix_path,root_node,receiver_node)
    print("混淆拓扑邻接矩阵转路由矩阵\n")
    run_ad_to_rout(confuse_adj_matrix_path,confuse_routing_matrix_path,root_node,receiver_node)

    print("---------------延迟矩阵提取相关延迟向量---------------")
    delay_vector_extractor=DelayMatrixProcessor(delay_matrix_path,delay_vector_path)
    delay_vector_extractor.extract_non_zero_delays()
    
    print("---------------操作矩阵求解准备---------------")
    
    copy_file(original_routing_matrix_path,M_path)
    copy_file(confuse_routing_matrix_path,F_path)
    copy_file(delay_vector_path,r_path)

    print("---------------数据准备完毕---------------")

def move_files_with_extensions(source_folder, target_folder, extensions):
    """
    将指定文件夹中具有特定扩展名的文件剪切到目标文件夹。
    如果目标文件夹中已存在同名文件，则覆盖它。

    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param extensions: 要剪切的文件扩展名列表，例如 ['.png', '.txt']
    """
    # 检查目标文件夹是否存在
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"目标文件夹 {target_folder} 不存在！")

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件扩展名是否在指定的扩展名列表中
        if any(filename.endswith(ext) for ext in extensions):
            # 构造源文件路径和目标文件路径
            source_file_path = os.path.join(source_folder, filename)
            target_file_path = os.path.join(target_folder, filename)

            # 剪切文件（如果目标文件已存在，将被覆盖）
            shutil.move(source_file_path, target_file_path)
            # print(f"文件 {filename} 已剪切到 {target_folder}（如果存在同名文件，则已覆盖）")

    # print("文件剪切完成！")

def move_data(topo_num):
    source_folder = "/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file"  # 源文件夹路径
    target_folder = os.path.join(source_folder, f"{topo_num}_output_file")  # 目标文件夹路径
    extensions = [".png", ".txt"]  # 需要剪切的文件扩展名列表

    try:
        move_files_with_extensions(source_folder, target_folder, extensions)
    except FileNotFoundError as e:
        print(e)

def vector_to_matrix(topo_num,prob_num):
    vector_path=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_output_file/{topo_num}_{prob_num}_deployment_vector.txt"
    matrix_path=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_output_file/{topo_num}_{prob_num}_deployment_matrix.txt"
    original_matrix_path=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/delay_result/{prob_num}/{topo_num}_simu_delay.txt"

    vector=np.loadtxt(vector_path)
    original_matrix=np.loadtxt(original_matrix_path)
    rows, cols=original_matrix.shape
    # print(f"rows:{rows}")
    matrix=vector_to_upper_triangle_matrix(vector=vector,n=rows)
    np.savetxt(matrix_path,matrix,fmt="%.6f")

def get_switch_num(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 遍历每一行，找到以switch_num:开头的行并提取数值
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith('switch_num:'):
                    # 分割冒号并提取右侧的数字，去除空格后转整数
                    return int(stripped_line.split(':')[1].strip())
        # 未找到对应行时返回None或抛出提示
        raise ValueError("文件中未找到'switch_num:'相关行")
    except Exception as e:
        print(f"读取失败：{e}")
        return None

def operation_solver_exe():
    input_file_dir="/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/input_file/"
    M_path=f"{input_file_dir}M.txt"
    F_path=f"{input_file_dir}F.txt"
    r_path=f"{input_file_dir}r.txt"

    #Recommended to be deprecated!!
    # run_test(M_path,F_path,r_path,method="scipy")
    # run_test(M_path,F_path,r_path,method="proximal")
    import time
    prob_num_arr = [500,1000,2000,3000,5000,7000,10000]
    time_arr = []
    topo_num=input("please input topo_num:")
    prepare_delay_matrix(topo_num=topo_num,prob_num_arr=prob_num_arr)

    node_num = get_switch_num(f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/{topo_num}_info.txt")
    
    for prob_num in prob_num_arr:
        prepare_data(topo_num,prob_num,M_path,F_path,r_path)
        # run_test(topo_num,prob_num,M_path,F_path,r_path,method="adam")
        # run_test(topo_num,prob_num,M_path,F_path,r_path,method="pgd")
        st = time.perf_counter()
        run_test(topo_num,prob_num,M_path,F_path,r_path,method="torch",node_num=node_num)
        dt = time.perf_counter()
        st = dt - st
        time_arr.append(st)
        print(f"cost time {st} s")
        move_data(topo_num=topo_num)
        vector_to_matrix(topo_num=topo_num,prob_num=prob_num)
        # break
    return sum(time_arr)/len(time_arr)

if __name__ == "__main__":
    st = operation_solver_exe()
    print(f"cost time {st} s")
    