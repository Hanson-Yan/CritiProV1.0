# torch_solver.py

import torch
import numpy as np
import matplotlib.pyplot as plt


class TorchOperationMatrixSolver:
    def __init__(self, topo_num, prob_num, gamma=0.01, eta=1, delta_max=100, epsilon=0.01, lr=0.01, max_iter=6000):
        """
        小规模建议（15节点数量以下）lr=0.01,max_iter=5000
        中等规模建议（15-30）lr=0.01,max_iter=6000
        大规模建议（30以上）lr=0.005,max_iter=1000
        """
        self.topo_num = topo_num
        self.prob_num = prob_num
        self.gamma = gamma
        self.eta = eta
        self.delta_max = delta_max
        self.epsilon = epsilon
        self.lr = lr
        self.max_iter = max_iter

        self.operation_matrix_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_operation_matrix.txt"
        self.deployment_vector_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_deployment_vector.txt"
        self.convergence_png_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file/{topo_num}_{prob_num}_convergence_curve.png"

    def read_matrix(self, file_path):
        return torch.tensor(np.loadtxt(file_path), dtype=torch.float32)

    def read_vector(self, file_path):
        return torch.tensor(np.loadtxt(file_path).reshape(-1, 1), dtype=torch.float32)

    def solve(self, M_file, F_file, r_file):
        M = self.read_matrix(M_file)
        F = self.read_matrix(F_file)
        r = self.read_vector(r_file)

        n = M.shape[0]
        P = torch.nn.Parameter(torch.rand((n, n), dtype=torch.float32), requires_grad=True)
        optimizer = torch.optim.Adam([P], lr=self.lr)

        loss_history = []

        for step in range(self.max_iter):
            optimizer.zero_grad()
            Pr = P @ r
            PM = P @ M

            # 构造各项损失
            loss_topo = torch.norm(PM - F, p=2)**2
            loss_delay = self.gamma * torch.norm(Pr - r, p=1)

            delta = Pr - r
            penalty = torch.sum(torch.relu(delta - self.delta_max)**2 + torch.relu(-delta)**2)
            row_norm = torch.sum(torch.relu(self.epsilon - torch.sum(torch.abs(P), dim=1)))

            total_loss = loss_topo + loss_delay + self.eta * penalty + row_norm
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                P.data = torch.clamp(P.data, 1e-3, 1.0)  # 元素范围
            loss_history.append(total_loss.item())

            if step % 20 == 0:
                print(f"Step {step}: Loss = {total_loss.item():.4f}")

        Pr_final = (P @ r).detach().numpy()
        np.savetxt(self.operation_matrix_path, P.detach().numpy(), fmt="%.6f")
        np.savetxt(self.deployment_vector_path, Pr_final, fmt="%.6f")

        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.title("Torch Optimization Convergence")
        plt.savefig(self.convergence_png_path)
        print(f"保存路径: {self.convergence_png_path}")
        return P.detach().numpy(), Pr_final
