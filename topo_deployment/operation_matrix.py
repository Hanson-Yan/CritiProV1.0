import numpy as np
from scipy.optimize import minimize

class OperationMatirxSolver:
    def __init__(self, gamma=0.1, eta=10, alpha=0.01, max_iter=1000, tol=1e-5):
        """
        初始化参数
        :param gamma: 稀疏性正则化参数
        :param eta: 罚函数惩罚因子
        :param alpha: 近端梯度法学习率
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        """
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.operation_matrix_path="/home/retr0/Project/TopologyObfu/CritiPro/output_file/operation_matrix.txt"
        self.deployment_vector_path="/home/retr0/Project/TopologyObfu/CritiPro/output_file/deployment_vector.txt"

    @staticmethod
    def read_matrix(file_path):
        """
        读取矩阵（M 或 F），文件每行是矩阵的一行，元素用空格分隔
        :param file_path: txt 文件路径
        :return: numpy 矩阵
        """
        return np.loadtxt(file_path, dtype=float)

    @staticmethod
    def read_vector(file_path):
        """
        读取延迟向量 r，每行是一个元素
        :param file_path: txt 文件路径
        :return: numpy 列向量
        """
        return np.loadtxt(file_path, dtype=float).reshape(-1, 1)

    def penalty_function(self, P, M, F, r, delta_max):
        """
        计算优化目标 H(F) = ||PM - F||_2 + γ||Pr - r||_1 + ηQ(P)
        """
        PM = P @ M
        Pr = P @ r
        term1 = np.linalg.norm(PM - F, ord=2)  # ||PM - F||_2
        term2 = self.gamma * np.linalg.norm(Pr - r, ord=1)  # γ||Pr - r||_1

        # 计算罚函数项
        delta = Pr - r
        term3 = np.sum(np.maximum(0, delta - delta_max) ** 2) + np.sum(np.maximum(0, -delta) ** 2)

        return term1 + term2 + self.eta * term3

    def optimize_P(self, M, F, r, delta_max=0.5):
        """
        使用 scipy.optimize 进行优化求解
        """
        n = M.shape[0]
        P_init = np.eye(n)  # 以单位矩阵为初始 P
        bounds = [(0, 1) for _ in range(n * n)]  # 约束 P 在 [0,1] 之间

        result = minimize(
            lambda P: self.penalty_function(P.reshape(n, n), M, F, r, delta_max)
                    + 0.05 * np.linalg.norm(P.reshape(n, n) - P_init, ord='fro'),  # 🔥 增加约束
            P_init.flatten(), method='L-BFGS-B', bounds=bounds
        )

        return result.x.reshape(n, n)

    @staticmethod
    def soft_thresholding(x, threshold):
        """
        软阈值操作: prox_{\gamma ||·||_1}(x)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


    def proximal_gradient_method(self, M, F, r, delta_max=100):
        """
        近端梯度法求解 P，支持完全自适应参数调整，并增加稀疏性控制
        """
        n = M.shape[0]
        P = np.random.rand(n, n)  # 采用随机初始化 P，让 P 更自然
        alpha = 0.01  #  适当增加 alpha 让优化步长更大
        gamma = 0.01  # 适当减少 gamma 让 P 不至于全 0
        eta = 0.5  # 适当增加 eta 让 P 不会陷入局部极小值
        prev_grad = None  # 记录前一次梯度变化

        # r = r / np.max(np.abs(r))  # 归一化 r，避免数值不稳定

        for iteration in range(self.max_iter):
            PM = P @ M
            Pr = P @ r

            # 计算梯度
            grad_P = 2 * (PM - F) @ M.T + gamma * np.sign(Pr - r) @ r.T

            # 计算罚函数梯度项
            # delta = Pr - r
            # penalty_grad = 2 * np.maximum(0, delta - delta_max) + 2 * np.maximum(0, -delta)
            # grad_P += eta * penalty_grad
            # 修正后的罚函数梯度计算
            delta = Pr - r
            # 计算条件项
            condition = np.zeros_like(delta)
            condition[delta > delta_max] = delta[delta > delta_max] - delta_max
            condition[delta < 0] = -delta[delta < 0]
            # 计算梯度项
            penalty_grad = 2 * self.eta * (condition @ r.T)  # (n,n)
            grad_P += penalty_grad

            max_grad = np.max(np.abs(grad_P))
            print(f"Iter {iteration}: max_grad={max_grad}, alpha={alpha}, eta={eta}")

            # 动态调整学习率
            if prev_grad is not None:
                if max_grad > prev_grad * 1.2:  # 梯度突然变大，降低学习率
                    alpha = max(alpha * 0.9, 0.005)
                elif max_grad < prev_grad * 0.8:  # 梯度下降太慢，增加学习率
                    alpha = min(alpha * 1.1, 0.1)
            
            prev_grad = max_grad

            # 梯度下降更新
            P_new = P - alpha * grad_P

            # 引入 `soft_thresholding` 让 P 保持稀疏
            # P_new = self.soft_thresholding(P_new, gamma)

            # 约束 P 在 [0,1] 之间
            P_new = np.clip(P_new, 0, 1)

            # 监测 P 是否过稀疏或接近单位矩阵
            if np.all(P_new < 1e-3):  
                eta = max(eta * 0.9, 0.05)  # P 变全 0，则减少 eta
            elif np.allclose(P_new, np.eye(n), atol=0.1):
                eta = min(eta * 1.1, 5)  # P 接近单位矩阵，则增大 eta

            # 检查收敛
            if np.linalg.norm(P_new - P) < self.tol:
                print(f" 迭代 {iteration} 次后收敛！")
                break

            P = P_new

        np.savetxt(self.operation_matrix_path, P, fmt="%.6f")
        print(f"生成的操作矩阵已保存至\n{self.operation_matrix_path}")
        np.savetxt(self.deployment_vector_path, Pr, fmt="%.6f")
        print(f"生成的部署向量已保存至\n{self.deployment_vector_path}")
        return P,Pr
    
    def solve(self, M_file, F_file, r_file, method="proximal", delta_max=0.5):
        """
        读取数据并求解 P
        :param M_file: 路由矩阵 txt 文件路径
        :param F_file: 混淆矩阵 txt 文件路径
        :param r_file: 延迟向量 txt 文件路径
        :param method: 选择优化方法 ("proximal" 或 "scipy")
        :param delta_max: 最大延迟约束
        :return: 求解出的 P 矩阵
        """
        M = self.read_matrix(M_file)
        F = self.read_matrix(F_file)
        r = self.read_vector(r_file)
        
        if method == "scipy":
            P = self.optimize_P(M, F, r, delta_max)
            return P,None
        elif method == "proximal":
            P, Pr= self.proximal_gradient_method(M, F, r, delta_max)
            return P,Pr
        else:
            return None
