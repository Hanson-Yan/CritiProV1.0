import numpy as np
import matplotlib.pyplot as plt
import os

class OperationMatrixAdamSolver:
    def __init__(self, topo_num,prob_num,gamma=0.0001, eta=1, alpha=0.01, max_iter=5000, tol=1e-4, delta_max=50, lambda_reg=0.1,lambda_grad=0.05, epsilon=0.01,deploy_reward=5):
        """
        初始化参数
        :param gamma: 稀疏性正则化参数
        :param eta: 罚函数惩罚因子
        :param alpha: 初始学习率
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :param delta_max: 最大延迟约束
        :param lambda_reg: 额外的正则化权重，保证P的每一行不全为0
        :param epsilon: 控制P每一行最小值的阈值
        """
        self.deploy_reward = deploy_reward
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.delta_max = delta_max
        self.lambda_reg = lambda_reg  # 额外的正则化项权重 
        self.lambda_grad = lambda_grad # 额外的保证Pr>=r梯度控制
        self.epsilon = epsilon  # 控制最小行范数阈值
        self.operation_matrix_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_operation_matrix.txt"
        self.deployment_vector_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        self.convergence_png_path=f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_convergence_curve.png"

    @staticmethod
    def read_matrix(file_path):
        return np.loadtxt(file_path, dtype=float)

    @staticmethod
    def read_vector(file_path):
        return np.loadtxt(file_path, dtype=float).reshape(-1, 1)

    def penalty_function(self, P, M, F, r):
        """
        计算优化目标 H(F) = ||PM - F||_2 + γ||Pr - r||_1 + ηQ(P) +  λ ∑ max(0, ε - ||P_i||_1) + λ2 ∑ max(0, r - Pr)^2
        """
        PM = P @ M
        Pr = P @ r
        term1 = np.linalg.norm(PM - F, ord=2)
        term2 = self.gamma * np.linalg.norm(Pr - r, ord=1)

        # 计算罚函数项
        delta = Pr - r
        term3 = np.sum(np.maximum(0, delta - self.delta_max) ** 2) + np.sum(np.maximum(0, -delta) ** 2)

        # 确保 P 每一行的 L1 范数不低于 ε
        term4 = np.sum(np.maximum(0, self.epsilon - np.sum(np.abs(P), axis=1)))

        # 额外约束 Pr >= r
        term5 = np.sum(np.maximum(0, r - Pr) ** 2)  # 让 Pr 尽可能 >= r

        return term1 + term2 + self.eta * term3 + self.lambda_reg * term4 + self.lambda_grad * term5
    
    # def penalty_function(self, P, M, F, r, iteration=0):
    #     PM = P @ M
    #     Pr = P @ r

    #     # 保持主逻辑不变，只调整数值范围
    #     SCALE_TERM1 = 100 / max(1, M.size)
    #     SCALE_TERM2 = 100 / max(1, r.size)

    #     term1 = np.linalg.norm(PM - F, ord=2) * SCALE_TERM1
    #     # term2 = self.gamma * np.linalg.norm(Pr - r, ord=1) * SCALE_TERM2
    #     # diff = np.abs(Pr - r)
    #     diff = Pr - r
    #     soft_mask = np.tanh((diff - 1.0) * 3)
    #     term2 = self.gamma * np.sum(np.maximum(0, soft_mask))

    #     delta = Pr - r
    #     term3 = (
    #         np.sum(np.maximum(0, delta - self.delta_max) ** 2) +
    #         np.sum(np.maximum(0, -delta) ** 2)
    #     )
    #     term3 = np.clip(term3, 0, 10)

    #     term4 = np.sum(np.maximum(0, self.epsilon - np.sum(np.abs(P), axis=1)))

    #     term5 = np.sum(np.maximum(0, r - Pr) ** 2)
    #     term5 = np.clip(term5, 0, 10)

    #     # 激励项：鼓励明显扰动（如 > 1）
    #     deployment_gain = np.sum(np.maximum(0, Pr - r - 1))  # 只有扰动超过 1 的部分才计入
    #     term6 = -self.deploy_reward * deployment_gain  # 负号是“奖励项”，越大越好
    #     return term1 + term2 + self.eta * term3 + self.lambda_reg * term4 + self.lambda_grad * term5 + term6


    def gradient(self, P, M, F, r):
        """
        计算目标函数对 P 的梯度
        """
        PM = P @ M
        Pr = P @ r
        grad_P = 2 * (PM - F) @ M.T + self.gamma * np.sign(Pr - r) @ r.T

        # 计算罚函数梯度项
        delta = Pr - r
        penalty_grad = 2 * np.maximum(0, delta - self.delta_max) + 2 * np.maximum(0, -delta)
        grad_P += self.eta * penalty_grad

        # 计算新正则项的梯度
        row_norms = np.sum(np.abs(P), axis=1, keepdims=True)
        grad_P += self.lambda_reg * (-1) * (row_norms < self.epsilon)

        # 对 Pr 低于 r 的部分施加梯度
        constraint_grad = -2 * np.maximum(0, r - Pr) @ r.T  # 只对 Pr < r 的部分生效
        grad_P += self.lambda_grad * constraint_grad  # 加入梯度

        return grad_P

    def adam_optimizer(self, M, F, r, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        使用 Adam 优化器进行优化，并确保 P 每一行至少有一个非零元素
        """
        n = M.shape[0]
        P = np.random.uniform(0.1, 1, size=(n, n))
        m = np.zeros_like(P)
        v = np.zeros_like(P)
        t = 0

        loss_values = []
        count_loss = 0
        for iteration in range(self.max_iter):
            t += 1
            grad_P = self.gradient(P, M, F, r)

            # Adam 公式
            m = beta1 * m + (1 - beta1) * grad_P
            v = beta2 * v + (1 - beta2) * grad_P ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # 更新 P
            P -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            P = np.clip(P, 1e-3, 1)  # 确保 P 在合理范围内
             # 记录每次迭代的目标函数值
            loss_value = self.penalty_function(P, M, F, r)
            loss_values.append(loss_value)
            if len(loss_values)>1 :
                if loss_values[-1]>loss_values[-2]:
                    print(f"-------------------learning rate 1/2--------------------")
                    lr=lr*0.5
                elif loss_values[-1]==loss_values[-2]:
                    count_loss+=1
                    if count_loss>10:
                        print(f"-------------------learning rate ++--------------------")
                        lr+=lr
                        count_loss=0
            # loss_values.append(self.penalty_function(P, M, F, r))
            # 打印每次迭代的信息
            if iteration % 10 == 0:  # 每10次迭代打印一次
                print(f"Iteration {iteration}, Objective Function Value: {loss_value:.6f}, Learning Rate: {lr:.6f}")

            if np.linalg.norm(grad_P) < self.tol:
                print(f"收敛于第 {iteration} 次迭代！")
                break

        # P = self.post_process_P(P)  # 进行后处理
        Pr = P @ r
        np.savetxt(self.operation_matrix_path, P, fmt="%.6f")
        print(f"生成的操作矩阵已保存至\n{self.operation_matrix_path}")
        np.savetxt(self.deployment_vector_path, Pr, fmt="%.6f")
        print(f"生成的部署向量已保存至\n{self.deployment_vector_path}")
        # 绘制收敛曲线
        plt.plot(loss_values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Convergence Curve')
        plt.grid(False)
        plt.savefig(self.convergence_png_path)  # 保存图像
        print(f"生成的收敛曲线已保存至\n{self.convergence_png_path}")
        # plt.show()
        return P, Pr

    # def enforce_pr_constraint(self,P, r, delta_min=1e-3):
    #     Pr = P @ r
    #     for i in range(P.shape[0]):
    #         if Pr[i] < r[i,0]:
    #             row = P[i, :].copy()
    #             current = row @ r[:,0]
    #             scale = (r[i,0] + delta_min) / current
    #             P[i, :] = np.clip(row * scale, 1e-3, 1)
    #     return P
    
    # def adam_optimizer(self, M, F, r, base_lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, warmup_steps=50):
    #     n = M.shape[0]

    #     # 学习率随规模自适应缩放
    #     scaled_lr = base_lr * (min(n, 100) / 100.0)

    #     # def get_lr(t, T_max):
    #     #     return max(1e-4, scaled_lr * (1 - t / T_max)) 
    #     def get_lr(t, max_iter, base_lr):
    #         return base_lr * (1 + np.cos(np.pi * t / max_iter)) / 2

    #     P = np.random.uniform(0.1, 1, size=(n, n))
    #     m = np.zeros_like(P)
    #     v = np.zeros_like(P)
    #     t = 0
    #     loss_values = []

    #     for iteration in range(self.max_iter):
    #         t += 1
    #         lr = get_lr(t, self.max_iter,self.alpha)

    #         grad_P = self.gradient(P, M, F, r)

    #         # Adam 更新
    #         m = beta1 * m + (1 - beta1) * grad_P
    #         v = beta2 * v + (1 - beta2) * grad_P ** 2
    #         m_hat = m / (1 - beta1 ** t)
    #         v_hat = v / (1 - beta2 ** t)

    #         P -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    #         # P = self.enforce_pr_constraint(P,r)

    #         P = np.clip(P, 1e-3, 1)

    #         # ✅ 传入当前迭代，启用动态正则项
    #         loss_value = self.penalty_function(P, M, F, r, iteration=iteration)
    #         loss_values.append(loss_value)

    #         if iteration % 10 == 0:
    #             print(f"Iteration {iteration}, Objective: {loss_value:.6f}, Learning Rate: {lr:.6f}")

    #         if np.linalg.norm(grad_P) < self.tol:
    #             print(f"收敛于第 {iteration} 次迭代！")
    #             break

    #     Pr = P @ r
    #     np.savetxt(self.operation_matrix_path, P, fmt="%.6f")
    #     print(f"操作矩阵保存至\n{self.operation_matrix_path}")
    #     np.savetxt(self.deployment_vector_path, Pr, fmt="%.6f")
    #     print(f"部署向量保存至\n{self.deployment_vector_path}")

    #     # 收敛曲线绘图
    #     plt.plot(loss_values)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Objective Function Value')
    #     plt.title('Convergence Curve')
    #     plt.grid(False)
    #     plt.savefig(self.convergence_png_path)
    #     print(f"收敛曲线已保存至\n{self.convergence_png_path}")

    #     return P, Pr

    # def post_process_P(self, P):
    #     """
    #     修正 P，防止某些行全为 0
    #     """
    #     for i in range(P.shape[0]):
    #         if np.all(P[i, :] == 0):  # 如果某行全 0
    #             P[i, i] = 1  # 在对角线上置 1

    #     return P



    def solve(self, M_file, F_file, r_file, method="adam"):
        """
        读取数据并求解 P
        """
        M = self.read_matrix(M_file)
        F = self.read_matrix(F_file)
        r = self.read_vector(r_file)

            # 🌟 拓扑规模归一化正则项（基准100个节点）
        n = M.shape[0]
        scale_factor = 100 / max(n, 1)
        self.gamma *= scale_factor
        self.lambda_reg *= scale_factor
        self.lambda_grad *= scale_factor

        if method == "adam":
            P, Pr = self.adam_optimizer(M, F, r, lr=self.alpha)
            # P, Pr = self.adam_optimizer(M, F, r, base_lr=self.alpha)
            return P, Pr
        else:
            return None
        # P, Pr = self.adam_optimizer(M, F, r)
        # return P, Pr
