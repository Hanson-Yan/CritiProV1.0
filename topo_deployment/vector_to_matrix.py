import numpy as np

def vector_to_upper_triangle_matrix(vector, n):
    """
    将给定向量转换为 n×n 的上三角矩阵（包括对角线）。
    
    参数:
        vector (list): 输入的向量，长度应为 n*(n+1)//2。
        n (int): 矩阵的行数和列数（方阵）。
    
    返回:
        numpy.ndarray: 生成的 n×n 上三角矩阵。
    """
    # 检查向量长度是否符合要求
    if len(vector) != n * (n - 1) // 2:
        raise ValueError("向量长度与方阵大小不匹配！")

    # 创建一个 n×n 的零矩阵
    matrix = np.zeros((n, n))

    # 填充矩阵
    index = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[i, j] = vector[index]
            index += 1

    return matrix

# 示例使用
# if __name__ == "__main__":
#     # 给定向量
#     vector = [12.04608, 12.798984, 12.604495, 12.918168, 27.0, 24.619193, 26.386514, 36.945024, 38.879733, 51.0]

#     # 用户输入矩阵的行数（假设是方阵）
#     n = int(input("请输入矩阵的行数（方阵）："))

#     try:
#         # 调用函数生成矩阵
#         matrix = vector_to_upper_triangle_matrix(vector, n)

#         # 打印矩阵
#         print("生成的矩阵为：")
#         for row in matrix:
#             print(" ".join(f"{num:.6f}" for num in row))
#     except ValueError as e:
#         print(e)