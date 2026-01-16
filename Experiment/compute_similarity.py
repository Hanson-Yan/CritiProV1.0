# graph_similarity.py
import sys
import numpy as np
utils_dir = "/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/utils"
sys.path.append(utils_dir)
from similarity import graph_similarity


# def matrix_levenshtein_distance(matrix1, matrix2):
#     """
#     计算两个矩阵之间的Levenshtein距离。
#     """
#     rows1, cols1 = matrix1.shape
#     rows2, cols2 = matrix2.shape

#     # 初始化距离矩阵
#     d = np.zeros((rows1 + 1, rows2 + 1), dtype=int)

#     # 填充距离矩阵
#     for i in range(1, rows1 + 1):
#         for j in range(1, rows2 + 1):
#             cost = 0 if np.array_equal(matrix1[i - 1, :], matrix2[j - 1, :]) else 1
#             d[i, j] = min(d[i - 1, j] + 1,      # 删除
#                           d[i, j - 1] + 1,      # 插入
#                           d[i - 1, j - 1] + cost)  # 替换

#     return d[rows1, rows2]

# def normalize_matrix(matrix):
#     """
#     归一化矩阵，使其值为0或1。
#     """
#     return (matrix > 0).astype(int)

# def graph_similarity(matrix1, matrix2):
#     """
#     计算两个图之间的相似性。
#     matrix1 和 matrix2 是图的邻接矩阵。
#     """
#     # 归一化矩阵
#     matrix1 = normalize_matrix(matrix1)
#     matrix2 = normalize_matrix(matrix2)

#     # 计算Levenshtein距离
#     LD = matrix_levenshtein_distance(matrix1, matrix2)
#     # print(f"LD:{LD}")
#     # 计算图的边数
#     L1 = np.sum(matrix1)
#     L2 = np.sum(matrix2)

#     # 计算相似性
#     if L1 == 0 and L2 == 0:
#         return 1.0  # 如果两个矩阵都没有边，相似性为1
#     elif L1 == 0 or L2 == 0:
#         return 0
    
#     score=1 - (LD / max(L1, L2))
#     if abs(L1 - L2)>25:#惩罚拓扑边数相差很大的拓扑
#         edge_penalty = abs(L1 - L2) / max(L1, L2)
#         # 加权惩罚（权重 alpha 可调）
#         alpha = 0.6
#         score = score * (1 - alpha * edge_penalty)
#     return score

def compute_similarity(topo_num):
    original_topo=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{topo_num}.txt")
    similarity_save_path=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/similarity_experiment/similarity_result.txt"
    model_name=["critipro","proto","antitomo"]
    similarity_arr=[]
    for name in model_name:
        _topo=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{name}_{topo_num}_confuse_topo.txt")
        topo_similarity=graph_similarity(original_topo, _topo, 'portrait')
        similarity_arr.append(topo_similarity)
    model_similarity={key: value for key, value in zip(model_name, similarity_arr)}
    print(model_similarity)
    with open(similarity_save_path, "w") as file:
        for key, value in model_similarity.items():
            file.write(f"{key}: {value}\n")


# def compute_similarity_test():
#     topo_num="topo_4"
#     name="antitomo"
#     original_topo=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{topo_num}.txt")
#     _topo=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{name}_{topo_num}_confuse_topo.txt")
#     topo_similarity=graph_similarity(original_topo,_topo)
#     print(f"topo_similarity:{topo_similarity}")
# compute_similarity_test()
# def compute_similarity():
#     # 示例图的邻接矩阵
#     data_path = "/home/retr0/Project/TopologyObfu/Experiment/data/"
#     result_path = "/home/retr0/Project/TopologyObfu/Experiment/result/similarity_experiment/"

#     original_matrix_file = "original_topo_matrix.txt"
#     antitomo_matrix_file = "antitomo_matrix_confuse.txt"
#     critipro_matrix_file = "critipro_matrix_confuse.txt"
#     proto_matrix_file = "proto_matrix_confuse.txt"

#     antitomo_similarity_file = result_path + antitomo_matrix_file.split("_")[0] + ".txt"
#     critipro_similarity_file = result_path + critipro_matrix_file.split("_")[0] + ".txt"
#     proto_similarity_file = result_path + proto_matrix_file.split("_")[0] + ".txt"
    
#     original_matrix = np.loadtxt(data_path + original_matrix_file)
#     antitomo_matrix = np.loadtxt(data_path + antitomo_matrix_file)
#     critipro_matrix = np.loadtxt(data_path + critipro_matrix_file)
#     proto_matrix = np.loadtxt(data_path + proto_matrix_file)


#     # 计算相似性
#     antitomo_similarity = graph_similarity(original_matrix, antitomo_matrix)
#     critipro_similarity = graph_similarity(original_matrix, critipro_matrix)
#     proto_similarity = graph_similarity(original_matrix, proto_matrix)
#     print(f"proto_similarity相似性: {proto_similarity}")
#     print(f"antitomo_similarity相似性: {antitomo_similarity}")
#     print(f"critipro_similarity相似性: {critipro_similarity}")
#     with open(antitomo_similarity_file, "w") as file:
#         file.write(f"{antitomo_similarity:.6f}\n")
#     with open(critipro_similarity_file, "w") as file:
#         file.write(f"{critipro_similarity:.6f}\n")
#     with open(proto_similarity_file, "w") as file:
#         file.write(f"{proto_similarity:.6f}\n")    

# if __name__ == "__main__":
#     compute_similarity()