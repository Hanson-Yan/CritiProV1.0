# node_delay_solver.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings


class NodeDelaySolver:
    """
    基于节点的延迟求解器（优化版）
    
    新增功能：
    1. 物理约束验证与修正（方案 A：物理下界）
    2. 结果验证诊断
    3. 早停机制
    4. 学习率调度
    
    物理模型：
    - r = M @ d，其中 d 是节点延迟向量
    - 优化目标：找到合理的节点延迟 d，使得预测延迟与实际测量接近
    """
    
    def __init__(self, topo_num, prob_num, unit_delay=12.0, lr=0.01, max_iter=3000,
                 early_stop_patience=100, early_stop_threshold=1e-5):
        """
        参数：
            topo_num: 拓扑编号
            prob_num: 探测次数
            unit_delay: 理论单位延迟（μs）
            lr: 初始学习率
            max_iter: 最大迭代次数
            early_stop_patience: 早停等待步数
            early_stop_threshold: 早停阈值（相对改进）
        """
        self.topo_num = topo_num
        self.prob_num = prob_num
        self.unit_delay = unit_delay
        self.lr = lr
        self.max_iter = max_iter
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        
        # 输出路径
        base_path = f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/output_file"
        self.node_delay_path = f"{base_path}/{topo_num}_{prob_num}_node_delays.txt"
        self.predicted_delay_path = f"{base_path}/{topo_num}_{prob_num}_predicted_delays.txt"
        self.convergence_png_path = f"{base_path}/{topo_num}_{prob_num}_node_delay_convergence.png"
        self.diagnostic_report_path = f"{base_path}/{topo_num}_{prob_num}_diagnostic_report.txt"
    
    def read_matrix(self, file_path):
        """读取矩阵文件"""
        return torch.tensor(np.loadtxt(file_path), dtype=torch.float32)
    
    def read_vector(self, file_path):
        """读取向量文件"""
        return torch.tensor(np.loadtxt(file_path).reshape(-1, 1), dtype=torch.float32)
    
    def validate_physical_constraints(self, M, r_measured):
        """
        物理约束验证与修正（方案 A：物理下界）
        
        检查每个接收器对的测量延迟是否 >= 物理下界
        物理下界 = 共享节点数 × unit_delay
        
        参数：
            M: 路由矩阵 (num_pairs, num_nodes)
            r_measured: 测量延迟向量 (num_pairs, 1)
            
        返回：
            corrected_delays: 修正后的延迟向量
            violations: 违反物理约束的配对索引
        """
        # 计算每对的共享节点数
        shared_counts = torch.sum(M, dim=1, keepdim=True)  # (num_pairs, 1)
        
        # 计算物理下界（方案 A）
        physical_lower_bound = shared_counts * self.unit_delay
        
        # 检查违反情况
        violations = (r_measured < physical_lower_bound).squeeze()
        num_violations = torch.sum(violations).item()
        
        # 修正：将小于物理下界的延迟提升到下界
        corrected_delays = torch.maximum(r_measured, physical_lower_bound)
        
        # 打印验证信息
        print(f"\n{'='*70}")
        print(f"物理约束验证（方案 A：物理下界）")
        print(f"{'='*70}")
        print(f"检查项目: 测量延迟 >= 共享节点数 × 单位延迟")
        print(f"单位延迟: {self.unit_delay} μs")
        print(f"总配对数: {len(r_measured)}")
        print(f"违反约束的配对数: {num_violations}")
        
        if num_violations > 0:
            print(f"\n⚠️  发现 {num_violations} 个配对的测量延迟低于物理下界")
            print(f"{'配对索引':<10} {'测量延迟(μs)':<15} {'物理下界(μs)':<15} {'修正后(μs)':<15}")
            print(f"{'-'*70}")
            
            violation_indices = torch.where(violations)[0]
            for idx in violation_indices:
                idx_val = idx.item()
                measured = r_measured[idx_val].item()
                lower = physical_lower_bound[idx_val].item()
                corrected = corrected_delays[idx_val].item()
                print(f"{idx_val:<10} {measured:>13.4f}  {lower:>13.4f}  {corrected:>13.4f}")
            
            print(f"\n✓ 已将这些配对的目标延迟修正为物理下界")
        else:
            print(f"✓ 所有配对的测量延迟均满足物理约束")
        
        print(f"{'='*70}\n")
        
        return corrected_delays, violation_indices if num_violations > 0 else None
    
    def should_early_stop(self, loss_history):
        """
        早停判断
        
        如果最近 patience 次迭代的损失相对改进 < threshold，则停止
        
        参数：
            loss_history: 损失历史列表
            
        返回：
            bool: 是否应该停止
        """
        if len(loss_history) < self.early_stop_patience:
            return False
        
        recent_losses = loss_history[-self.early_stop_patience:]
        
        # 计算相对改进
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        
        if max_loss == 0:
            return True
        
        relative_improvement = (max_loss - min_loss) / max_loss
        
        return relative_improvement < self.early_stop_threshold
    
    def solve(self, M_file, r_file):
        """
        求解节点延迟
        
        参数：
            M_file: 路由矩阵文件路径
            r_file: 测量延迟向量文件路径
            
        返回：
            d: 优化后的节点延迟向量
            r_pred: 预测的延迟向量
        """
        # 读取数据
        M = self.read_matrix(M_file)  # (num_pairs, num_nodes)
        r_measured = self.read_vector(r_file)  # (num_pairs, 1)
        
        num_nodes = M.shape[1]
        num_pairs = M.shape[0]
        
        # ========== 步骤 1: 物理约束验证与修正 ==========
        r_target, violations = self.validate_physical_constraints(M, r_measured)
        
        # 初始化节点延迟（接近理论值）
        d = torch.nn.Parameter(
            torch.full((num_nodes, 1), self.unit_delay, dtype=torch.float32),
            requires_grad=True
        )
        
        # 优化器
        optimizer = torch.optim.Adam([d], lr=self.lr)
        
        # ========== 步骤 2: 学习率调度器 ==========
        # 当损失不再下降时，降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,          # 学习率衰减因子
            patience=50,         # 等待步数
            verbose=True,
            min_lr=1e-6          # 最小学习率
        )
        
        # 记录损失
        loss_history = []
        
        # 打印优化开始信息
        print(f"\n{'='*70}")
        print(f"开始优化节点延迟 (拓扑 {self.topo_num}, 探测 {self.prob_num})")
        print(f"{'='*70}")
        print(f"路由矩阵形状: {M.shape}")
        print(f"测量延迟向量长度: {num_pairs}")
        print(f"节点数量: {num_nodes}")
        print(f"理论单位延迟: {self.unit_delay} μs")
        print(f"初始学习率: {self.lr}")
        print(f"最大迭代次数: {self.max_iter}")
        print(f"早停等待步数: {self.early_stop_patience}")
        print(f"{'='*70}\n")
        
        # ========== 步骤 3: 优化循环（带早停） ==========
        early_stopped = False
        final_step = self.max_iter
        
        for step in range(self.max_iter):
            optimizer.zero_grad()
            
            # 计算预测延迟
            r_pred = M @ d  # (num_pairs, 1)
            
            # 损失函数
            # 1. 预测误差（主要目标，使用修正后的目标）
            loss_fit = torch.mean((r_pred - r_target) ** 2)
            
            # 2. 正则化：节点延迟应接近理论值
            loss_reg = 0.01 * torch.mean((d - self.unit_delay) ** 2)
            
            # 3. 约束：节点延迟应为正且在合理范围内
            loss_constraint = torch.sum(torch.relu(-d)) + \
                             torch.sum(torch.relu(d - self.unit_delay * 5))
            
            # 总损失
            total_loss = loss_fit + loss_reg + loss_constraint
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 投影到可行域
            with torch.no_grad():
                d.data = torch.clamp(d.data, 0.1, self.unit_delay * 5)
            
            # 记录损失
            loss_history.append(total_loss.item())
            
            # 更新学习率
            scheduler.step(total_loss)
            
            # 打印进度
            if step % 100 == 0:
                rmse = torch.sqrt(loss_fit).item()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:4d}: Loss = {total_loss.item():.6f}, "
                      f"RMSE = {rmse:.4f} μs, LR = {current_lr:.6f}")
            
            # ========== 早停检查 ==========
            if self.should_early_stop(loss_history):
                print(f"\n🛑 早停触发于第 {step} 步")
                print(f"   原因: 最近 {self.early_stop_patience} 步损失改进 < {self.early_stop_threshold}")
                early_stopped = True
                final_step = step
                break
        
        # 最终结果
        d_final = d.detach().numpy()
        r_pred_final = (M @ d).detach().numpy()
        
        # ========== 步骤 4: 结果验证与诊断 ==========
        diagnostic_info = self.validate_solution(
            d_final, M.numpy(), r_target.numpy(), r_measured.numpy(), r_pred_final
        )
        
        # 计算最终误差
        errors = r_measured.numpy() - r_pred_final
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        
        print(f"\n{'='*70}")
        print(f"优化完成")
        print(f"{'='*70}")
        print(f"实际迭代次数: {final_step}")
        if early_stopped:
            print(f"早停状态: 是（节省 {self.max_iter - final_step} 次迭代）")
        else:
            print(f"早停状态: 否（达到最大迭代次数）")
        print(f"最终 RMSE: {rmse:.4f} μs")
        print(f"平均绝对误差: {mae:.4f} μs")
        print(f"最大绝对误差: {np.max(np.abs(errors)):.4f} μs")
        print(f"{'='*70}\n")
        
        # 保存结果
        np.savetxt(self.node_delay_path, d_final, fmt="%.6f")
        np.savetxt(self.predicted_delay_path, r_pred_final, fmt="%.6f")
        print(f"节点延迟已保存到: {self.node_delay_path}")
        print(f"预测延迟已保存到: {self.predicted_delay_path}")
        
        # 保存诊断报告
        self._save_diagnostic_report(diagnostic_info, early_stopped, final_step)
        
        # 绘制收敛曲线
        self._plot_convergence(loss_history)
        
        # 打印节点延迟详情
        self._print_node_delays(d_final)
        
        return d_final, r_pred_final
    
    def validate_solution(self, d, M, r_target, r_measured, r_pred):
        """
        结果验证与诊断
        
        检查项：
        1. 节点延迟是否在合理范围
        2. 预测延迟误差是否可接受
        3. 是否存在异常节点
        4. 物理约束满足情况
        
        参数：
            d: 节点延迟向量
            M: 路由矩阵
            r_target: 目标延迟（修正后）
            r_measured: 原始测量延迟
            r_pred: 预测延迟
            
        返回：
            dict: 诊断信息
        """
        warnings_list = []
        
        print(f"\n{'='*70}")
        print(f"结果验证与诊断")
        print(f"{'='*70}\n")
        
        # ========== 检查 1: 节点延迟范围 ==========
        d_flat = d.flatten()
        min_delay = 0.5 * self.unit_delay
        max_delay = 5.0 * self.unit_delay
        
        outliers_low = np.where(d_flat < min_delay)[0]
        outliers_high = np.where(d_flat > max_delay)[0]
        
        print(f"[检查 1] 节点延迟范围")
        print(f"  合理范围: [{min_delay:.2f}, {max_delay:.2f}] μs")
        print(f"  实际范围: [{np.min(d_flat):.2f}, {np.max(d_flat):.2f}] μs")
        
        if len(outliers_low) > 0:
            warnings_list.append(f"⚠️  {len(outliers_low)} 个节点延迟过低 (< {min_delay:.2f} μs): {outliers_low}")
            print(f"  ⚠️  过低节点: {outliers_low}")
        
        if len(outliers_high) > 0:
            warnings_list.append(f"⚠️  {len(outliers_high)} 个节点延迟过高 (> {max_delay:.2f} μs): {outliers_high}")
            print(f"  ⚠️  过高节点: {outliers_high}")
        
        if len(outliers_low) == 0 and len(outliers_high) == 0:
            print(f"  ✓ 所有节点延迟在合理范围内")
        
        # ========== 检查 2: 预测误差 ==========
        errors = r_measured.flatten() - r_pred.flatten()
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        print(f"\n[检查 2] 预测延迟误差")
        print(f"  RMSE: {rmse:.4f} μs")
        print(f"  MAE: {mae:.4f} μs")
        print(f"  最大误差: {max_error:.4f} μs")
        
        error_threshold = 5.0  # 5 μs 阈值
        if rmse > error_threshold:
            warnings_list.append(f"⚠️  RMSE ({rmse:.2f} μs) 超过阈值 {error_threshold} μs")
            print(f"  ⚠️  RMSE 超过阈值")
        else:
            print(f"  ✓ 预测误差在可接受范围内")
        
        # ========== 检查 3: 异常配对 ==========
        large_errors = np.where(np.abs(errors) > 10.0)[0]  # 误差 > 10 μs 的配对
        
        print(f"\n[检查 3] 异常配对（误差 > 10 μs）")
        if len(large_errors) > 0:
            warnings_list.append(f"⚠️  {len(large_errors)} 个配对的预测误差过大")
            print(f"  发现 {len(large_errors)} 个异常配对:")
            print(f"  {'配对索引':<10} {'测量(μs)':<12} {'预测(μs)':<12} {'误差(μs)':<12}")
            print(f"  {'-'*50}")
            for idx in large_errors[:5]:  # 只显示前 5 个
                print(f"  {idx:<10} {r_measured[idx, 0]:>10.4f}  {r_pred[idx, 0]:>10.4f}  {errors[idx]:>10.4f}")
            if len(large_errors) > 5:
                print(f"  ... 还有 {len(large_errors) - 5} 个（详见报告文件）")
        else:
            print(f"  ✓ 无异常配对")
        
        # ========== 检查 4: 物理约束满足情况 ==========
        shared_counts = np.sum(M, axis=1, keepdims=True)
        physical_lower = shared_counts * self.unit_delay
        violations = r_pred < physical_lower * 0.95  # 允许 5% 误差
        
        print(f"\n[检查 4] 物理约束满足情况")
        num_violations = np.sum(violations)
        if num_violations > 0:
            warnings_list.append(f"⚠️  {num_violations} 个配对的预测延迟低于物理下界")
            print(f"  ⚠️  {num_violations} 个配对违反物理约束")
        else:
            print(f"  ✓ 所有预测延迟满足物理约束")
        
        # ========== 总结 ==========
        print(f"\n{'='*70}")
        if len(warnings_list) == 0:
            print(f"✅ 验证通过：未发现明显问题")
        else:
            print(f"⚠️  发现 {len(warnings_list)} 个潜在问题:")
            for i, warning in enumerate(warnings_list, 1):
                print(f"  {i}. {warning}")
        print(f"{'='*70}\n")
        
        return {
            'warnings': warnings_list,
            'outliers_low': outliers_low,
            'outliers_high': outliers_high,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'large_error_pairs': large_errors,
            'constraint_violations': np.where(violations)[0]
        }
    
    def _save_diagnostic_report(self, diagnostic_info, early_stopped, final_step):
        """保存诊断报告到文件"""
        with open(self.diagnostic_report_path, 'w', encoding='utf-8') as f:
            f.write(f"节点延迟求解诊断报告\n")
            f.write(f"{'='*70}\n")
            f.write(f"拓扑编号: {self.topo_num}\n")
            f.write(f"探测次数: {self.prob_num}\n")
            f.write(f"优化完成步数: {final_step}\n")
            f.write(f"早停状态: {'是' if early_stopped else '否'}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"验证结果\n")
            f.write(f"{'='*70}\n\n")
            
            f.write(f"RMSE: {diagnostic_info['rmse']:.4f} μs\n")
            f.write(f"MAE: {diagnostic_info['mae']:.4f} μs\n")
            f.write(f"最大误差: {diagnostic_info['max_error']:.4f} μs\n\n")
            
            if len(diagnostic_info['warnings']) > 0:
                f.write(f"警告信息:\n")
                for i, warning in enumerate(diagnostic_info['warnings'], 1):
                    f.write(f"  {i}. {warning}\n")
            else:
                f.write(f"✓ 无警告\n")
            
            f.write(f"\n{'='*70}\n")
            f.write(f"详细信息\n")
            f.write(f"{'='*70}\n\n")
            
            if len(diagnostic_info['outliers_low']) > 0:
                f.write(f"低延迟异常节点: {diagnostic_info['outliers_low'].tolist()}\n")
            
            if len(diagnostic_info['outliers_high']) > 0:
                f.write(f"高延迟异常节点: {diagnostic_info['outliers_high'].tolist()}\n")
            
            if len(diagnostic_info['large_error_pairs']) > 0:
                f.write(f"\n大误差配对 (误差 > 10 μs):\n")
                for idx in diagnostic_info['large_error_pairs']:
                    f.write(f"  配对 {idx}\n")
        
        print(f"诊断报告已保存到: {self.diagnostic_report_path}")
    
    def _plot_convergence(self, loss_history):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制损失曲线
        plt.plot(loss_history, linewidth=2, color='#2E86AB')
        plt.xlabel("Iteration", fontsize=14, fontweight='bold')
        plt.ylabel("Loss", fontsize=14, fontweight='bold')
        plt.title(f"Node Delay Optimization Convergence\n(Topo {self.topo_num}, Prob {self.prob_num})", 
                  fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.yscale('log')
        
        # 标注最终损失
        final_loss = loss_history[-1]
        plt.axhline(y=final_loss, color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(len(loss_history) * 0.7, final_loss * 1.5, 
                 f'Final Loss: {final_loss:.6f}', 
                 fontsize=12, color='red')
        
        plt.tight_layout()
        plt.savefig(self.convergence_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"收敛曲线已保存到: {self.convergence_png_path}")
    
    def _print_node_delays(self, d):
        """打印节点延迟详情"""
        print(f"\n{'='*70}")
        print(f"节点延迟详情")
        print(f"{'='*70}")
        print(f"{'节点':<8} {'延迟(μs)':<14} {'与理论值差异(μs)':<22} {'状态':<10}")
        print(f"{'-'*70}")
        
        for i, delay in enumerate(d.flatten()):
            diff = delay - self.unit_delay
            
            # 判断状态
            if delay < 0.5 * self.unit_delay:
                status = "⚠️  过低"
            elif delay > 5.0 * self.unit_delay:
                status = "⚠️  过高"
            else:
                status = "✓ 正常"
            
            print(f"{i:<8} {delay:>12.4f}    {diff:>18.4f}      {status:<10}")
        
        print(f"{'='*70}\n")


# 测试代码
if __name__ == "__main__":
    # 测试路径
    M_file = "/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/input_file/M.txt"
    r_file = "/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment/data/input_file/r.txt"
    
    solver = NodeDelaySolver(
        topo_num="topo_1",
        prob_num=500,
        unit_delay=12.0,
        lr=0.01,
        max_iter=3000,
        early_stop_patience=100,
        early_stop_threshold=1e-5
    )
    
    d, r_pred = solver.solve(M_file, r_file)
    
    print("\n" + "="*70)
    print("优化后的节点延迟:")
    print("="*70)
    print(d)
    print("\n" + "="*70)
    print("预测的延迟向量:")
    print("="*70)
    print(r_pred)
