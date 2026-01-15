import numpy as np
import cvxpy as cp

class ProTO:
    def __init__(self, A, x, delta_max):
        self.A = A  # 路由矩阵 A ∈ ℝ^{m×n}
        self.x = x.flatten()  # 路由延迟向量 x ∈ ℝ^m
        self.delta_max = delta_max
        self.m, self.n = A.shape

        assert len(self.x) == self.m, "延迟向量x的维度必须与路由矩阵A的行数一致"

    def solve_optimization(self, A_m):
        """
        求解优化问题：
        min_F ||F A - A_m||_F^2
        subject to: 0 <= F x - x <= δ_max
        """
        # 决策变量：混淆矩阵 F ∈ ℝ^{m×m}
        F = cp.Variable((self.m, self.m))

        # 构造目标函数： Frobenius 范数的平方
        objective = cp.Minimize(cp.norm(F @ self.A - A_m, 'fro') ** 2)

        # 构造约束条件： 0 ≤ F x − x ≤ δ_max
        Fx = F @ self.x
        constraints = [
            Fx - self.x >= 0,
            Fx - self.x <= self.delta_max
        ]

        # 构造并求解优化问题
        prob = cp.Problem(objective, constraints)
        # prob.solve(solver=cp.OSQP)  # 可换成 SCS 或 ECOS 等
        prob.solve()  # 可换成 SCS 或 ECOS 等

        if F.value is None:
            raise ValueError("优化求解失败，返回值为空。可能是约束过紧或矩阵奇异。")

        return F.value  # 返回混淆矩阵 F 的数值解

    def compute_Fx(self, F):
        """
        计算 F @ x，即混淆后的延迟向量
        """
        return F @ self.x

# # 测试用例
# m, n = 3, 3  # 简化问题规模
# A_real = np.random.rand(m, n)
# x = np.random.rand(m)
# delta_max = 0.5
# A_fake = np.random.rand(m, n)

# # 初始化求解
# proto = ProTO(A_real, x, delta_max)
# F = proto.solve_optimization(A_fake)
# Fx = proto.compute_Fx(F)
# print(Fx)
# # 验证约束满足情况
# print("约束违反检查:")
# print("Max upper violation:", np.max(F @ x - (x + delta_max)))  # 应 <= 0
# print("Max lower violation:", np.max(x - F @ x))                # 应 <= 0