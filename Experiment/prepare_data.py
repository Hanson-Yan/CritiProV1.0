import os
import sys
from pathlib import Path


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


def prepare_data(topo_num):
    topo_num_result=topo_num+"_result/"

    data_dir = Path("/home/retr0/Project/TopologyObfu/Experiment/data/")
    files_in_data = {
    "original_topo_matrix": data_dir / "original_topo_matrix.txt",
    "original_delay_vector": data_dir / "original_delay_vector.txt",
    "critipro_matrix_confuse": data_dir / "critipro_matrix_confuse.txt",
    "critipro_deployment_vector": data_dir / "critipro_deployment_vector.txt",
    "proto_matrix_confuse": data_dir / "proto_matrix_confuse.txt",
    "proto_deployment_vector": data_dir / "proto_deployment_vector.txt",
    "antitomo_matrix_confuse": data_dir / "antitomo_matrix_confuse.txt",
    "antitomo_deployment_vector": data_dir / "antitomo_deployment_vector.txt"
}

    # 检查所有文件是否存在
    missing_files = [file_name for file_name, file_path in files_in_data.items() if not file_path.exists()]
    if missing_files:
        print(f"The following files do not exist: {', '.join(missing_files)}")
        sys.exit(1)

    print("all in data")
    
    #original file
    critipro_dir="/home/retr0/Project/TopologyObfu/CritiPro/"
    original_matrix_txt=topo_num+".txt"
    delay_vector_txt="output_file/delay_vector.txt"
    original_matrix_path=critipro_dir+topo_num_result+original_matrix_txt
    delay_vector_path=critipro_dir+topo_num_result+delay_vector_txt
    if os.path.exists(original_matrix_path):
        copy_file(original_matrix_path,files_in_data["original_topo_matrix"])
    else:
        print(f"original_matrix_path is not exist")
        sys.exit(1)

    if os.path.exists(delay_vector_path):
        copy_file(delay_vector_path,files_in_data["original_delay_vector"])
    else:
        print(f"delay matrix is not exist")
        sys.exit(1)
    

    #critipro file
    topo_matrix_confuse_txt="output_file/topo_matrix_confuse.txt"
    deployment_vector_txt="output_file/deployment_vector.txt"
    critipro_confuse_matrix_path=critipro_dir+topo_num_result+topo_matrix_confuse_txt
    critipro_deployment_vector_path=critipro_dir+topo_num_result+deployment_vector_txt

    if os.path.exists(critipro_confuse_matrix_path):
        copy_file(critipro_confuse_matrix_path,files_in_data["critipro_matrix_confuse"])
    else:
        print(f"{critipro_confuse_matrix_path} is not exist")
        sys.exit(1)

    if os.path.exists(critipro_deployment_vector_path):
        copy_file(critipro_deployment_vector_path,files_in_data["critipro_deployment_vector"])
    else:
        print(f"{critipro_deployment_vector_path} is not exist")
        sys.exit(1)

    #proto file
    proto_dir="/home/retr0/Project/TopologyObfu/ProTO/"
    topo_matrix_confuse_txt="output_file/topo_matrix_confuse.txt"
    deployment_vector_txt="output_file/deploy_vector.txt"
    proto_confuse_matrix_path=proto_dir+topo_num_result+topo_matrix_confuse_txt
    proto_deployment_vector_path=proto_dir+topo_num_result+deployment_vector_txt

    if os.path.exists(proto_confuse_matrix_path):
        copy_file(proto_confuse_matrix_path,files_in_data["proto_matrix_confuse"])
    else:
        print(f"{proto_confuse_matrix_path} is not exist")
        sys.exit(1)

    if os.path.exists(proto_deployment_vector_path):
        copy_file(proto_deployment_vector_path,files_in_data["proto_deployment_vector"])
    else:
        print(f"{proto_deployment_vector_path} is not exist")
        sys.exit(1)
    
    #antitomo file
    antitomo_dir="/home/retr0/Project/TopologyObfu/AntiTomo/"
    topo_matrix_confuse_txt="output_file/topo_matrix_confuse.txt"
    deployment_vector_txt="output_file/deploy_vector.txt"
    antitomo_confuse_matrix_path=antitomo_dir+topo_num_result+topo_matrix_confuse_txt
    antitomo_deployment_vector_path=antitomo_dir+topo_num_result+deployment_vector_txt

    if os.path.exists(antitomo_confuse_matrix_path):
        copy_file(antitomo_confuse_matrix_path,files_in_data["antitomo_matrix_confuse"])
    else:
        print(f"{antitomo_confuse_matrix_path} is not exist")
        sys.exit(1)

    if os.path.exists(antitomo_deployment_vector_path):
        copy_file(antitomo_deployment_vector_path,files_in_data["antitomo_deployment_vector"])
    else:
        print(f"{antitomo_deployment_vector_path} is not exist")
        sys.exit(1)


def prepare_topo_data(topo_num):
    #original topo tree
    experiment_topo_data_dir=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/"
    try:
        os.makedirs(experiment_topo_data_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        print(f"文件夹 {experiment_topo_data_dir} 已创建成功！")
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")

    # mininet_topo_tree=f"/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree/{topo_num}.txt"
    # copy_file(mininet_topo_tree,f"{experiment_topo_data_dir}{topo_num}.txt")
    mininet_topo_matrix=f"/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/{topo_num}.txt"
    copy_file(mininet_topo_matrix,f"{experiment_topo_data_dir}{topo_num}.txt")

    #critipro confuse topo
    critipro_confuse_topo=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/data/{topo_num}_output_file/output_adj.txt"
    copy_file(critipro_confuse_topo,f"{experiment_topo_data_dir}critipro_{topo_num}_confuse_topo.txt")

    #proto confuse topo
    proto_confuse_topo=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/topo_matrix_confuse.txt"
    copy_file(proto_confuse_topo,f"{experiment_topo_data_dir}proto_{topo_num}_confuse_topo.txt")

    #antitomo_confuse_topo
    antitomo_confuse_topo=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/topo_matrix_confuse.txt"
    copy_file(antitomo_confuse_topo,f"{experiment_topo_data_dir}antitomo_{topo_num}_confuse_topo.txt")


def prepare_delay_data(topo_num):
    experiment_delay_data_dir=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/delay"

    #original
    original_delay_data_dir=f"{experiment_delay_data_dir}/original"
    try:
        os.makedirs(original_delay_data_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        print(f"文件夹 {original_delay_data_dir} 已创建成功！")
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")
    prob_num_arr=[500,1000,2000,3000,5000,7000,10000]    
    
    for prob_num in prob_num_arr:
        delay_file=f"/home/retr0/Project/TopologyObfu/CritiPro/{topo_num}_result/delay_result/{prob_num}/{topo_num}_delay_vector.txt"
        delay_file_ex=f"{original_delay_data_dir}/{topo_num}_{prob_num}_delay.txt"
        copy_file(delay_file,delay_file_ex)

    #critipro
    critipro_delay_data_dir=f"{experiment_delay_data_dir}/critipro"
    try:
        os.makedirs(critipro_delay_data_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        print(f"文件夹 {critipro_delay_data_dir} 已创建成功！")
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")
    prob_num_arr=[500,1000,2000,3000,5000,7000,10000]    
    
    for prob_num in prob_num_arr:
        delay_file=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        delay_file_ex=f"{critipro_delay_data_dir}/{topo_num}_{prob_num}_delay.txt"
        copy_file(delay_file,delay_file_ex)

    #proto
    proto_delay_data_dir=f"{experiment_delay_data_dir}/proto"
    try:
        os.makedirs(proto_delay_data_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        print(f"文件夹 {proto_delay_data_dir} 已创建成功！")
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")
    prob_num_arr=[500,1000,2000,3000,5000,7000,10000]    
    
    for prob_num in prob_num_arr:
        delay_file=f"/home/retr0/Project/TopologyObfu/ProTO/{topo_num}_result/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        delay_file_ex=f"{proto_delay_data_dir}/{topo_num}_{prob_num}_delay.txt"
        copy_file(delay_file,delay_file_ex)

    #antitomo
    antitomo_delay_data_dir=f"{experiment_delay_data_dir}/antitomo"
    try:
        os.makedirs(antitomo_delay_data_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        print(f"文件夹 {antitomo_delay_data_dir} 已创建成功！")
    except Exception as e:
        print(f"创建文件夹时发生错误：{e}")
    prob_num_arr=[500,1000,2000,3000,5000,7000,10000]    
    
    for prob_num in prob_num_arr:
        delay_file=f"/home/retr0/Project/TopologyObfu/AntiTomo/{topo_num}_result/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        delay_file_ex=f"{antitomo_delay_data_dir}/{topo_num}_{prob_num}_delay.txt"
        copy_file(delay_file,delay_file_ex)



# if __name__=="__main__":
#     prepare_data(topo_num)