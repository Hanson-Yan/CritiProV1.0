import numpy as np
import matplotlib.pyplot as plt
import os

class PGDOperationMatrixSolver:
    def __init__(self, topo_num, prob_num, gamma=0.01, alpha=0.01, max_iter=3000, tol=1e-4, delta_max=50, epsilon=0.01):
        self.topo_num = topo_num
        self.prob_num = prob_num
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.delta_max = delta_max
        self.epsilon = epsilon

        self.operation_matrix_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_operation_matrix.txt"
        self.deployment_vector_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        self.convergence_png_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_convergence_curve.png"

    @staticmethod
    def read_matrix(file_path):
        return np.loadtxt(file_path, dtype=float)

    @staticmethod
    def read_vector(file_path):
        return np.loadtxt(file_path, dtype=float).reshape(-1, 1)

    def objective(self, P, M, F, r):
        PM = P @ M
        Pr = P @ r
        return np.linalg.norm(PM - F)**2 + self.gamma * np.linalg.norm(Pr - r, 1)

    def gradient(self, P, M, F, r):
        PM = P @ M
        Pr = P @ r
        return 2 * (PM - F) @ M.T + self.gamma * np.sign(Pr - r) @ r.T

    def project_constraints(self, P, r):
        # 保证 r 是列向量
        if r.ndim == 1:
            r = r.reshape(-1, 1)

        # 投影 P 元素范围 [1e-3, 1]
        P = np.clip(P, 1e-3, 1.0)

        # 保证每一行的 L1 ≥ epsilon
        row_sums = np.sum(np.abs(P), axis=1, keepdims=True)
        scale = np.maximum(self.epsilon / (row_sums + 1e-6), 1.0)
        P = P * scale

        # 保证 Pr - r ∈ [0, δ_max]
        Pr = P @ r
        delta = np.clip(Pr - r, 0, self.delta_max)

        # 用最小二乘反推 P（近似地让 P @ r ≈ r + delta）
        # P = (r + delta) @ np.linalg.pinv(r.T)
        P = (r + delta) @ r.T @ np.linalg.pinv(r @ r.T)

        return np.clip(P, 1e-3, 1.0)
    


    def solve(self, M_file, F_file, r_file, method="pgd"):
        M = self.read_matrix(M_file)
        F = self.read_matrix(F_file)
        r = self.read_vector(r_file)

        n = M.shape[0]
        scale_factor = 100 / max(n, 1)
        self.gamma *= scale_factor

        P = np.random.uniform(0.01, 1.0, size=(n, n))
        loss_history = []

        for i in range(self.max_iter):
            grad = self.gradient(P, M, F, r)
            P -= self.alpha * grad
            P = self.project_constraints(P, r)

            loss = self.objective(P, M, F, r)
            loss_history.append(loss)

            if i % 20 == 0:
                print(f"Iter {i}, Loss: {loss:.4f}")

            if np.linalg.norm(grad) < self.tol:
                print(f"Converged at iteration {i}")
                break

        Pr = P @ r

        # 保存结果
        np.savetxt(self.operation_matrix_path, P, fmt="%.6f")
        print(f"操作矩阵已保存到：{self.operation_matrix_path}")
        np.savetxt(self.deployment_vector_path, Pr, fmt="%.6f")
        print(f"部署向量已保存到：{self.deployment_vector_path}")

        # 保存收敛图
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("PGD Convergence")
        plt.grid(True)
        plt.savefig(self.convergence_png_path)
        print(f"收敛曲线图已保存到：{self.convergence_png_path}")

        return P, Pr

# import numpy as np
# import matplotlib.pyplot as plt
# import os

# class PGDOperationMatrixSolver:
#     def __init__(self, topo_num, prob_num, 
#                  gamma=0.001, 
#                  alpha=0.1, 
#                  max_iter=3000, 
#                  tol=1e-5,
#                  delta_max=100, 
#                  epsilon=0.05,
#                  momentum=0.9,
#                  decay_step=200):
#         # 初始化参数与文件路径（保持不变）
#         self.topo_num = topo_num
#         self.prob_num = prob_num
#         self.gamma = gamma
#         self.alpha = alpha
#         self.max_iter = max_iter
#         self.tol = tol
#         self.delta_max = delta_max
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.decay_step = decay_step

#         # 文件路径配置（根据实际需求修改）
#         self.operation_matrix_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_operation_matrix.txt"
#         self.deployment_vector_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
#         self.convergence_png_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_convergence_curve.png"


#     @staticmethod
#     def read_matrix(file_path):
#         return np.loadtxt(file_path, dtype=float)

#     @staticmethod
#     def read_vector(file_path):
#         v = np.loadtxt(file_path, dtype=float)
#         return v.reshape(-1, 1)  # 确保列向量

#     def objective(self, P, M, F, r):
#         PM = P @ M
#         Pr = P @ r
#         return (np.linalg.norm(PM - F)**2 
#                + self.gamma * np.sum(np.abs(Pr - r)))

#     def gradient(self, P, M, F, r):
#         PM = P @ M
#         Pr = P @ r
#         return 2 * (PM - F) @ M.T + self.gamma * np.sign(Pr - r) @ r.T

#     def project_constraints(self, P, r):
#         # 维度校验与转换
#         if r.ndim == 1:
#             r = r.reshape(-1, 1)
#         assert r.shape[1] == 1, "输入r必须是列向量"
        
#         # 阶段1：元素范围投影
#         P = np.clip(P, 1e-3, 1.0)
        
#         # 阶段2：行范数平滑处理
#         row_sums = np.sum(P, axis=1, keepdims=True)
#         under_rows = row_sums < self.epsilon
#         P += under_rows * (self.epsilon - row_sums) / P.shape[1]
        
#         # 阶段3：Pr约束精确投影
#         for _ in range(3):
#             Pr = P @ r
#             delta = np.clip(Pr - r, 0, self.delta_max)
            
#             # 关键修正：正确的矩阵运算维度
#             denominator = r.T @ r
#             if denominator < 1e-8:
#                 denominator = 1e-8
#             P = (r + delta) @ r.T / denominator
            
#             # 重新应用元素约束
#             P = np.clip(P, 1e-3, 1.0)
        
#         return P

#     def solve(self, M_path, F_path, r_path):
#         # 数据读取
#         M = self.read_matrix(M_path)
#         F = self.read_matrix(F_path)
#         r = self.read_vector(r_path)  # 确保列向量
        
#         # 智能初始化
#         n = M.shape[0]
#         P = np.clip(F @ np.linalg.pinv(M), 1e-3, 1.0) * 0.8 
#         P += 0.2 * np.random.uniform(0.01, 0.1, (n, n))
        
#         # 优化循环
#         velocity = np.zeros_like(P)
#         loss_history = []
        
#         for iter in range(self.max_iter):
#             # 学习率衰减
#             if iter % self.decay_step == 0 and iter > 0:
#                 self.alpha *= 0.8
#                 print(f"学习率衰减至: {self.alpha:.4f}")
            
#             # 梯度计算与动量更新
#             grad = self.gradient(P, M, F, r)
#             velocity = self.momentum * velocity + grad
#             P -= self.alpha * velocity
            
#             # 投影到约束空间
#             P = self.project_constraints(P, r)
            
#             # 记录损失
#             loss = self.objective(P, M, F, r)
#             loss_history.append(loss)
            
#             # 打印进度
#             if iter % 50 == 0:
#                 grad_norm = np.linalg.norm(grad)
#                 avg_delay = np.mean(P @ r - r)
#                 print(f"Iter {iter:4d} | Loss: {loss:.2e} | Grad: {grad_norm:.2e} | ΔDelay: {avg_delay:.2f}ms")
                
#             # 早停机制
#             if iter > 100 and (loss_history[-100] - loss) < self.tol:
#                 print(f"早停于第{iter}次迭代")
#                 break
        
#         # 保存结果
#         np.savetxt(self.operation_matrix_path, P, fmt="%.6f")
#         np.savetxt(self.deployment_vector_path, P @ r, fmt="%.6f")
        
#         # 绘制收敛曲线
#         plt.figure()
#         plt.semilogy(loss_history)
#         plt.xlabel("Iteration"), plt.ylabel("Log Loss")
#         plt.title("PGD Convergence")
#         plt.savefig(self.convergence_png_path)
#         plt.close()
        
#         return P, P @ r

