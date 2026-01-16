import os
import numpy as np

data_path="/home/retr0/Project/TopologyObfu/MininetTop/probeCode/host_result/"
def extract_delay_and_variance(num_hosts, base_path=data_path):
    """
    提取每个主机文件中的相对延迟数据和方差数据，并整理成矩阵。

    :param num_hosts: 主机数量
    :param base_path: 存储主机文件的基路径
    :return: 延迟矩阵和方差矩阵
    """
    # 初始化延迟矩阵和方差矩阵
    delay_matrix = np.zeros((num_hosts, num_hosts))
    variance_matrix = np.zeros((num_hosts, num_hosts))

    # 读取每个主机的文件并提取数据
    for i in range(1, num_hosts + 1):  # 主机序号从 h1 开始
        host_file = os.path.join(base_path, f"h{i}_results.txt")
        if os.path.exists(host_file):
            with open(host_file, "r") as f:
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if j >= num_hosts - i:
                        break
                    parts = line.split(", ")
                    delay = float(parts[0].split(": ")[1].strip(" ms"))
                    variance = float(parts[1].split(": ")[1].strip())
                    delay_matrix[i-1, i + j] = delay  # 调整索引
                    variance_matrix[i-1, i + j] = variance  # 调整索引

    return delay_matrix, variance_matrix

def save_matrices(delay_matrix, variance_matrix, delay_path="delay_matrix.txt", variance_path="variance_matrix.txt"):
    """
    将延迟矩阵和方差矩阵保存到文件。

    :param delay_matrix: 延迟矩阵
    :param variance_matrix: 方差矩阵
    :param delay_path: 延迟矩阵保存路径
    :param variance_path: 方差矩阵保存路径
    """
    delay_file = os.path.join(data_path, delay_path)
    variance_file = os.path.join(data_path, variance_path)
    np.savetxt(delay_file, delay_matrix, fmt="%.6f")
    print(f"delay matrix have saved in \n{delay_file}")
    np.savetxt(variance_file, variance_matrix, fmt="%.6f")
    print(f"variance matrix have saved in \n{variance_file}")

# 示例调用
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: sudo /home/retr0/anaconda3/bin/python ./extract_delay_data.py num_hosts")
        sys.exit(1)
    num_hosts = int(sys.argv[1])  # 根据你的网络拓扑调整
    delay_matrix, variance_matrix = extract_delay_and_variance(num_hosts)
    
    # 打印提取的延迟矩阵和方差矩阵
    print("Delay Matrix:")
    print(delay_matrix)
    print("Variance Matrix:")
    print(variance_matrix)

    # 保存矩阵到文件
    save_matrices(delay_matrix, variance_matrix)