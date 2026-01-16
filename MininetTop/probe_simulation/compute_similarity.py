# import numpy as np

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

#     # 计算图的边数
#     L1 = np.sum(matrix1)
#     L2 = np.sum(matrix2)

#     # 计算相似性
#     if L1 == 0 and L2 == 0:
#         return 1.0  # 如果两个矩阵都没有边，相似性为1
#     elif L1 == 0 or L2 == 0:
#         return 0
#     return 1 - (LD / max(L1, L2))

# def compute_similarity(topo_num):
#     # 示例图的邻接矩阵
#     original_matrix_file = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/topo_tree_adj/"+topo_num+".txt"
#     infer_matrix_file = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/infer_topo/"+topo_num+"_infer.txt"

#     original_matrix = np.loadtxt(original_matrix_file)
#     infer_matrix = np.loadtxt(infer_matrix_file)
#     # 计算相似性
#     similarity = graph_similarity(original_matrix, infer_matrix)
#     print(similarity)
# if __name__ == "__main__":
#     compute_similarity()

import numpy as np
from scipy.stats import pearsonr
class MatrixSimilarityEvaluator:
    def __init__(self, real_matrix, measured_matrix, amplify_power=5.5, divergence_alpha=6.0):
        self.real_matrix = self._extract_upper_triangle(real_matrix)
        self.measured_matrix = self._extract_upper_triangle(measured_matrix)
        self.amplify_power = amplify_power
        self.divergence_alpha = divergence_alpha

        self._apply_amplified_error()

    def _extract_upper_triangle(self, matrix):
        return matrix[np.triu_indices_from(matrix, k=0)]

    def _apply_amplified_error(self):
        """非线性误差放大处理，差距越大惩罚越重"""
        error = self.measured_matrix - self.real_matrix
        amplified_error = np.sign(error) * (np.abs(error) ** self.amplify_power)
        self.measured_matrix = self.real_matrix + amplified_error

    def compute_mse(self):
        return np.mean((self.real_matrix - self.measured_matrix) ** 2)

    def compute_normalized_mse(self):
        numerator = np.sum((self.real_matrix - self.measured_matrix) ** 2)
        denominator = np.sum(self.real_matrix ** 2) + 1e-8
        return numerator / denominator

    def compute_similarity_score(self):
        """引入误差对数放大因子，增强对小误差的敏感度"""
        error = np.abs(self.real_matrix - self.measured_matrix)
        amplified = np.log1p(error) ** 2  # log(1 + x) 放大小误差，再平方加剧影响
        normalized = amplified / (np.mean(self.real_matrix) + 1e-6)
        penalty = np.mean(normalized)
        return max(0.0, 1.0 - penalty)

    def compute_pearson_correlation(self):
        corr, _ = pearsonr(self.real_matrix, self.measured_matrix)
        return corr

    def compute_divergence_score(self):
        """采用高阶次幂的归一化差异"""
        diff = np.abs(self.real_matrix - self.measured_matrix) ** self.divergence_alpha
        base = np.abs(self.real_matrix) ** self.divergence_alpha + 1e-8
        score = 1.0 - (np.sum(diff) / np.sum(base))
        return score

    def summary(self):
        return {
            "MSE": self.compute_mse(),
            "Normalized MSE": self.compute_normalized_mse(),
            "Similarity Score (Custom Log+Power)": self.compute_similarity_score(),
            "Pearson Correlation": self.compute_pearson_correlation(),
            f"Divergence Score (α={self.divergence_alpha})": self.compute_divergence_score()
        }
    def smilarity_score(self):
        # print(f"Similarity Score (Custom Log+Power): {self.compute_similarity_score()}") 
        return self.compute_similarity_score()
    

def exe_similarity(topo_num,probe_num):
    simu_delay=f"/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/{probe_num}/{topo_num}_simu_delay.txt"
    base_delay=f"/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/delay_result/{topo_num}_base_delay.txt"
    real_matrix=np.loadtxt(base_delay)
    measured_matrix=np.loadtxt(simu_delay)
    exe = MatrixSimilarityEvaluator(real_matrix, measured_matrix)
    similarity_score=exe.smilarity_score()
    # print(similarity_score)
    return similarity_score