# similarity.py - 修正版

import numpy as np
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.stats import entropy
import scipy.sparse
# ==========================
# 方法一：Network Portrait Divergence (推荐)
# ==========================
# 论文来源: Bagrow & Bollt, "An information-theoretic, all-scales approach to comparing networks" (2019)
def get_portrait(G):
    """
    计算图的 Portrait 矩阵 B
    B[l, k] = 网络中在距离 l 处拥有 k 个节点的起始节点数量
    """
    # 获取所有节点对的最短路径长度
    # 对于大图，这步是主要耗时点，但比同构判定快得多
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # 找到最大直径 (max_dist) 和最大节点数 (N)
    max_dist = 0
    for src in path_lengths:
        max_dist = max(max_dist, max(path_lengths[src].values()))
    
    N = G.number_of_nodes()
    
    # 初始化 Portrait 矩阵
    # 行：距离 l (0 到 diameter)
    # 列：节点数 k (0 到 N)
    B = np.zeros((max_dist + 1, N + 1))
    
    for src in path_lengths:
        # 统计当前节点 src 在各个距离上有多少个邻居
        # dists 是一个列表，如 [0, 1, 1, 2, 2, 2...] 表示距离为0的有1个，距离1的有2个...
        distances = list(path_lengths[src].values())
        
        # 统计每个距离出现的次数 (即 shell size)
        # dist_counts[d] = 距离 d 的节点数量
        dist_counts = {}
        for d in distances:
            dist_counts[d] = dist_counts.get(d, 0) + 1
            
        # 填充矩阵
        for d, count in dist_counts.items():
            B[d, count] += 1
            
    return B
def pad_portraits(B1, B2):
    """对齐两个 Portrait 矩阵的大小"""
    r1, c1 = B1.shape
    r2, c2 = B2.shape
    
    r_max = max(r1, r2)
    c_max = max(c1, c2)
    
    P1 = np.zeros((r_max, c_max))
    P2 = np.zeros((r_max, c_max))
    
    P1[:r1, :c1] = B1
    P2[:r2, :c2] = B2
    
    return P1, P2
def graph_similarity_portrait(matrix1, matrix2):
    """
    基于 Network Portrait Divergence 的相似度
    返回值: 0.0 (完全不同) ~ 1.0 (完全相同)
    """
    # 转换为 NetworkX 图 (NPD 需要 BFS/最短路径，用 NX 实现最高效)
    G1 = nx.from_numpy_array(matrix1)
    G2 = nx.from_numpy_array(matrix2)
    
    # 1. 计算肖像矩阵
    B1 = get_portrait(G1)
    B2 = get_portrait(G2)
    
    # 2. 对齐矩阵维度
    B1, B2 = pad_portraits(B1, B2)
    
    # 3. 计算概率分布
    # 行归一化：P(k | l) = 在距离 l 处有 k 个节点的概率
    # 我们不仅要比较分布，还要考虑距离 l 的权重
    
    # 这里使用论文中的加权方法构建扁平化分布向量
    # P(k, l) = P(k|l) * P(l)
    
    # 总行数 (距离)
    num_rows = B1.shape[0]
    
    # 辅助向量
    # row_sums: 每个距离 l 涉及的总节点路径数
    row_sums1 = np.sum(B1, axis=1)
    row_sums2 = np.sum(B2, axis=1)
    
    # 防止除零
    row_sums1[row_sums1 == 0] = 1
    row_sums2[row_sums2 == 0] = 1
    
    # 概率矩阵 P 和 Q
    P = B1 / row_sums1[:, None] # P(k|l)
    Q = B2 / row_sums2[:, None]
    
    # 距离分布 P(l) = 该距离层包含的路径总数 / 总路径数
    # 实际上，原始论文为了简化，直接比较加权的分布
    # 这里使用一个简化的 Information Theoretic 比较：
    
    # 将矩阵拉平为分布向量进行比较
    # 为了保留距离 l 的信息，我们对每一行进行加权
    # 简单的做法：直接拉平并归一化为概率分布
    p_vec = B1.flatten()
    q_vec = B2.flatten()
    
    p_vec = p_vec / np.sum(p_vec)
    q_vec = q_vec / np.sum(q_vec)
    
    # 4. 计算 Jensen-Shannon 散度 (JSD)
    # JSD 范围是 0 (相同) 到 1 (最大差异, log2 base)
    M = 0.5 * (p_vec + q_vec)
    js_divergence = 0.5 * (entropy(p_vec, M, base=2) + entropy(q_vec, M, base=2))
    
    # 转换为相似度
    return 1.0 - js_divergence

def get_laplacian_eigenvalues(matrix):
    """
    计算拉普拉斯矩阵的特征值
    Laplacian L = D - A
    """
    # 确保矩阵是对称的
    matrix = (matrix > 0).astype(float)
    
    # 计算度矩阵 D
    degrees = np.sum(matrix, axis=1)
    D = np.diag(degrees)
    
    # 拉普拉斯矩阵 L
    L = D - matrix
    
    # 计算特征值 (使用 eigvalsh 因为 L 是实对称矩阵，计算更快更稳)
    # 只需要特征值，不需要特征向量
    eigenvalues = np.linalg.eigvalsh(L)
    
    # 从小到大排序 (eigvalsh 通常已排序，但为了保险)
    return np.sort(eigenvalues)
def graph_similarity_spectral(matrix1, matrix2):
    """
    基于拉普拉斯谱距离的相似度 (0~1)
    优点：不受节点编号顺序影响 (Isomorphism Invariant)
    """
    n1 = matrix1.shape[0]
    n2 = matrix2.shape[0]
    
    # 1. 对齐维度：如果大小不一样，用0填充较小的图，使其特征值数量一致
    n = max(n1, n2)
    
    # 获取特征值 k (k=n)
    eig1 = get_laplacian_eigenvalues(matrix1)
    eig2 = get_laplacian_eigenvalues(matrix2)
    
    # 补齐特征值向量 (用0补齐)
    if n1 < n:
        eig1 = np.pad(eig1, (0, n - n1), 'constant')
    if n2 < n:
        eig2 = np.pad(eig2, (0, n - n2), 'constant')
        
    # 2. 计算欧几里得距离 (Euclidean Distance)
    dist = np.linalg.norm(eig1 - eig2)
    
    # 3. 归一化距离转相似度
    # 归一化因子是一个经验值，通常取决于节点数量
    # 这里使用简单的指数衰减将距离映射到 0-1
    # 或者使用 1 / (1 + dist)
    similarity = 1 / (1 + dist)
    
    return similarity
# ==========================
# 方法一：旧版 Levenshtein（适用于小拓扑）
# ==========================

def matrix_levenshtein_distance(matrix1, matrix2):
    """
    计算两个矩阵之间的Levenshtein距离。
    """
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    d = np.zeros((rows1 + 1, rows2 + 1), dtype=int)

    for i in range(1, rows1 + 1):
        for j in range(1, rows2 + 1):
            cost = 0 if np.array_equal(matrix1[i - 1, :], matrix2[j - 1, :]) else 1
            d[i, j] = min(d[i - 1, j] + 1,
                          d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)

    return d[rows1, rows2]

def normalize_matrix(matrix):
    """归一化矩阵"""
    return (matrix > 0).astype(int)

def graph_similarity_leve(matrix1, matrix2):
    """
    旧版相似性（适用于小拓扑 n<50）
    """
    matrix1 = normalize_matrix(matrix1)
    matrix2 = normalize_matrix(matrix2)

    LD = matrix_levenshtein_distance(matrix1, matrix2)
    L1 = np.sum(matrix1)
    L2 = np.sum(matrix2)

    if L1 == 0 and L2 == 0:
        return 1.0
    elif L1 == 0 or L2 == 0:
        return 0.0
    
    return 1 - (LD / max(L1, L2))

# ==========================
# 方法二：边 Jaccard 相似度（推荐）
# ==========================

def graph_similarity_jaccard(matrix1, matrix2):
    """
    基于边集合的 Jaccard 相似度（最直观，O(m)）
    适用于任意规模拓扑
    """
    matrix1 = normalize_matrix(matrix1)
    matrix2 = normalize_matrix(matrix2)
    
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    
    # 两个图节点数必须相同
    if n1 != n2:
        # 扩展较小的矩阵
        n = max(n1, n2)
        if n1 < n:
            new_mat1 = np.zeros((n, n), dtype=int)
            new_mat1[:n1, :n1] = matrix1
            matrix1 = new_mat1
        if n2 < n:
            new_mat2 = np.zeros((n, n), dtype=int)
            new_mat2[:n2, :n2] = matrix2
            matrix2 = new_mat2
    
    n = matrix1.shape[0]
    
    # 提取边集合
    edges1 = set()
    edges2 = set()
    
    for i in range(n):
        for j in range(i+1, n):
            if matrix1[i, j] > 0:
                edges1.add((i, j))
            if matrix2[i, j] > 0:
                edges2.add((i, j))
    
    # Jaccard 相似度
    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    
    if union == 0:
        return 1.0
    
    return intersection / union

# ==========================
# 方法三：度数分布相似度（快速但不精确）
# ==========================

def graph_similarity_degree(matrix1, matrix2):
    """
    基于度数分布的相似性（O(n)）
    适用于大拓扑的快速估计
    """
    matrix1 = normalize_matrix(matrix1)
    matrix2 = normalize_matrix(matrix2)
    
    # 计算度数分布
    degree_dist1 = defaultdict(int)
    degree_dist2 = defaultdict(int)
    
    for row in matrix1:
        degree = np.sum(row)
        degree_dist1[degree] += 1
    
    for row in matrix2:
        degree = np.sum(row)
        degree_dist2[degree] += 1
    
    # 计算分布差异
    all_degrees = set(degree_dist1.keys()).union(set(degree_dist2.keys()))
    total_diff = sum(abs(degree_dist1.get(d, 0) - degree_dist2.get(d, 0)) for d in all_degrees)
    
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    max_diff = n1 + n2  # 最大可能差异
    
    return 1 - (total_diff / max_diff)

def edge_edit_distance(matrix1, matrix2):
    """
    真正的边编辑距离：计算边的增删改次数
    """
    # 归一化
    matrix1 = (matrix1 > 0).astype(int)
    matrix2 = (matrix2 > 0).astype(int)
    
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    
    # 确保矩阵大小相同
    if n1 != n2:
        n = max(n1, n2)
        if n1 < n:
            new_mat1 = np.zeros((n, n), dtype=int)
            new_mat1[:n1, :n1] = matrix1
            matrix1 = new_mat1
        if n2 < n:
            new_mat2 = np.zeros((n, n), dtype=int)
            new_mat2[:n2, :n2] = matrix2
            matrix2 = new_mat2
    
    n = matrix1.shape[0]
    
    # 计算边差异（只看上三角，避免重复计数）
    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matrix1[i, j] != matrix2[i, j]:
                distance += 1  # 边的增加或删除
    
    return distance
def graph_similarity_edge_based(matrix1, matrix2):
    """
    基于边编辑距离的相似度（符合论文描述）
    """
    matrix1 = (matrix1 > 0).astype(int)
    matrix2 = (matrix2 > 0).astype(int)
    
    # 计算边编辑距离
    distance = edge_edit_distance(matrix1, matrix2)
    
    # 计算边数（注意除以2，因为邻接矩阵对称）
    edge_count1 = int(np.sum(matrix1) / 2)
    edge_count2 = int(np.sum(matrix2) / 2)
    
    max_edges = max(edge_count1, edge_count2)
    
    if max_edges == 0:
        return 1.0
    
    # 相似度 = 1 - (编辑距离 / 最大边数)
    similarity = 1 - (distance / max_edges)
    
    return similarity



# ==========================
# 统一接口
# ==========================

def graph_similarity(matrix1, matrix2, method='leve'):
    """
    统一的图相似度计算接口
    
    参数:
        method: 'leve'（旧版）, 'jaccard'（推荐）, 'degree'（快速）
    """
    if method == 'leve':
        return graph_similarity_leve(matrix1, matrix2)
    elif method == 'portrait':
        return graph_similarity_portrait(matrix1, matrix2)
    elif method == 'spectral':
        return graph_similarity_spectral(matrix1, matrix2)
    elif method == 'edge':
        return graph_similarity_edge_based(matrix1, matrix2)
    elif method == 'jaccard':
        return graph_similarity_jaccard(matrix1, matrix2)
    elif method == 'degree':
        return graph_similarity_degree(matrix1, matrix2)
    else:
        # 默认：小拓扑用 Jaccard，大拓扑用度数
        n = matrix1.shape[0]
        if n <= 30:
            return graph_similarity_jaccard(matrix1, matrix2)
        else:
            return graph_similarity_degree(matrix1, matrix2)


if __name__ == "__main__":
    # 1. 4x4 网格 (16节点)
    G1 = nx.grid_2d_graph(4, 4)
    M1 = nx.to_numpy_array(G1)
    
    # 2. 5x5 网格 (25节点)
    G2 = nx.grid_2d_graph(5, 5)
    M2 = nx.to_numpy_array(G2)
    
    # 3. 随机图 (16节点，边数和G1差不多)
    G3 = nx.gnm_random_graph(16, G1.number_of_edges())
    M3 = nx.to_numpy_array(G3)
    
    print("--- 形状相似性测试 ---")
    print(f"Grid(4x4) vs Grid(5x5) [形状应相似]:")
    print(f"  Portrait Sim: {graph_similarity(M1, M2, 'portrait'):.4f} (预期: 高分)")
    # print(f"  NetSimile Sim: {graph_similarity(M1, M2, 'netsimile'):.4f}")
    
    print(f"\nGrid(4x4) vs Random [形状应不同]:")
    print(f"  Portrait Sim: {graph_similarity(M1, M3, 'portrait'):.4f} (预期: 低分)")
    # print(f"  NetSimile Sim: {graph_similarity(M1, M3, 'netsimile'):.4f}")