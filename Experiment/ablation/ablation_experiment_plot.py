# ablation_experiment_plot.py - 修改版（图例简化为 Random vs. CritiPro）

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import os

# 设置论文级别的字体和样式
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 16


class ComparisonPlotter:
    def __init__(self, results_json_path):
        """
        初始化绘图器
        :param results_json_path: comparison_results.json 的路径
        """
        if not os.path.exists(results_json_path):
            raise FileNotFoundError(f"结果文件不存在: {results_json_path}")
        
        with open(results_json_path, 'r') as f:
            self.results = json.load(f)
        
        # 转换为 DataFrame
        self.df = pd.DataFrame(self.results)
        
        # 只保留成功的实验
        self.df = self.df[self.df['status'] == 'success']
        
        # 获取输出目录
        self.output_dir = os.path.dirname(results_json_path)
        
        print(f"加载了 {len(self.df)} 条成功的实验记录")
        print(f"输出目录: {self.output_dir}")
        
        # 打印数据预览
        print("\n数据预览:")
        print(self.df[['topo_num', 'constraint_type', 'fitting_error_mse', 
                       'unrealizable_ratio', 'deployment_cost_l1']].head(10))
    
    def plot_mse_comparison(self, filename='comparison_mse.png', 
                           figsize=(10, 6), dpi=300):
        """
        绘制 MSE 对比图（独立图片）
        
        :param filename: 输出文件名
        :param figsize: 图片尺寸
        :param dpi: 分辨率
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 按拓扑和方法分组，计算平均值
        grouped_mse = self.df.groupby(['topo_num', 'constraint_type'])['fitting_error_mse'].mean().unstack()
        
        # 重命名列（简化为 Random 和 CritiPro）
        column_mapping = {
            'random': 'Random',
            'ours': 'CritiPro'
        }
        grouped_mse = grouped_mse.rename(columns=column_mapping)
        
        # 调整列顺序
        column_order = ['Random', 'CritiPro']
        grouped_mse = grouped_mse[column_order]
        
        # 颜色设置
        colors = ['#E74C3C', '#3498DB']  # 红色（Random）、蓝色（CritiPro）
        
        x = np.arange(len(grouped_mse.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, grouped_mse.iloc[:, 0], width,
                      label=grouped_mse.columns[0], color=colors[0],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        bars2 = ax.bar(x + width/2, grouped_mse.iloc[:, 1], width,
                      label=grouped_mse.columns[1], color=colors[1],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Topology', fontweight='bold')
        ax.set_ylabel('Deployment Fitting Error (MSE)', fontweight='bold')
        ax.set_title('Topology Fitting Error Comparison', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('topo_', 'Topo ') for t in grouped_mse.index])
        
        # 使用对数坐标（如果数值差异大）
        max_mse = grouped_mse.max().max()
        min_mse = grouped_mse.min().min()
        
        if max_mse / min_mse > 100:  # 如果差异超过100倍，使用对数坐标
            ax.set_yscale('log')
            ax.set_ylabel('Deployment Fitting Error (MSE, log scale)', fontweight='bold')
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        
        # 添加数值标签
        def add_value_labels(bars, use_log=False):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if use_log or height > 1000:
                        label = f'{height:.1e}'  # 科学计数法
                    else:
                        label = f'{height:.1f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1, use_log=(max_mse / min_mse > 100))
        add_value_labels(bars2, use_log=(max_mse / min_mse > 100))
        
        plt.tight_layout()
        
        # 保存高分辨率 PNG
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        print(f"✓ 图表已保存: {output_path} (DPI={dpi})")
        plt.close()
    
    def plot_unrealizable_comparison(self, filename='comparison_unrealizable.png',
                                     figsize=(10, 6), dpi=300):
        """
        绘制 Unrealizable Ratio 对比图（独立图片）
        
        :param filename: 输出文件名
        :param figsize: 图片尺寸
        :param dpi: 分辨率
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 按拓扑和方法分组
        grouped_unreal = self.df.groupby(['topo_num', 'constraint_type'])['unrealizable_ratio'].mean().unstack()
        
        # 重命名列（简化为 Random 和 CritiPro）
        column_mapping = {
            'random': 'Random',
            'ours': 'CritiPro'
        }
        grouped_unreal = grouped_unreal.rename(columns=column_mapping)
        
        # 调整列顺序
        column_order = ['Random', 'CritiPro']
        grouped_unreal = grouped_unreal[column_order]
        
        # 颜色设置
        colors = ['#E74C3C', '#3498DB']
        
        x = np.arange(len(grouped_unreal.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, grouped_unreal.iloc[:, 0], width,
                      label=grouped_unreal.columns[0], color=colors[0],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        bars2 = ax.bar(x + width/2, grouped_unreal.iloc[:, 1], width,
                      label=grouped_unreal.columns[1], color=colors[1],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Topology', fontweight='bold')
        ax.set_ylabel('Unrealizable Virtual Link Ratio (%)', fontweight='bold')
        # ax.set_title('Physical Deployability Comparison', 
                    # fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('topo_', 'Topo ') for t in grouped_unreal.index])
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存高分辨率 PNG
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        print(f"✓ 图表已保存: {output_path} (DPI={dpi})")
        plt.close()
    
    def plot_deployment_cost(self, filename='comparison_deployment_cost.png',
                            figsize=(10, 6), dpi=300):
        """
        绘制 Deployment Cost (L1) 对比图
        
        :param filename: 输出文件名
        :param figsize: 图片尺寸
        :param dpi: 分辨率
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 按拓扑和方法分组
        grouped = self.df.groupby(['topo_num', 'constraint_type'])['deployment_cost_l1'].mean().unstack()
        
        # 重命名列（简化为 Random 和 CritiPro）
        column_mapping = {
            'random': 'Random',
            'ours': 'CritiPro'
        }
        grouped = grouped.rename(columns=column_mapping)
        
        # 调整列顺序
        column_order = ['Random', 'CritiPro']
        grouped = grouped[column_order]
        
        # 颜色设置
        colors = ['#E74C3C', '#3498DB']
        
        x = np.arange(len(grouped.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, grouped.iloc[:, 0], width,
                      label=grouped.columns[0], color=colors[0],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        bars2 = ax.bar(x + width/2, grouped.iloc[:, 1], width,
                      label=grouped.columns[1], color=colors[1],
                      edgecolor='black', linewidth=1.2, alpha=0.85)
        
        ax.set_xlabel('Topology', fontweight='bold')
        ax.set_ylabel('Deployment Cost (L1 Norm)', fontweight='bold')
        ax.set_title('Deployment Cost Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('topo_', 'Topo ') for t in grouped.index])
        
        # 检查是否需要对数坐标
        max_cost = grouped.max().max()
        min_cost = grouped.min().min()
        
        if max_cost / min_cost > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Deployment Cost (L1 Norm, log scale)', fontweight='bold')
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        
        # 添加数值标签
        def add_value_labels(bars, use_log=False):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if use_log or height > 1000:
                        label = f'{height:.1e}'
                    else:
                        label = f'{height:.1f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1, use_log=(max_cost / min_cost > 100))
        add_value_labels(bars2, use_log=(max_cost / min_cost > 100))
        
        plt.tight_layout()
        
        # 保存高分辨率 PNG
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        print(f"✓ 图表已保存: {output_path} (DPI={dpi})")
        plt.close()
    
    def print_summary_statistics(self):
        """
        打印统计摘要
        """
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        
        for topo in sorted(self.df['topo_num'].unique()):
            print(f"\n{topo.upper()}:")
            df_topo = self.df[self.df['topo_num'] == topo]
            
            for method in ['random', 'ours']:
                df_subset = df_topo[df_topo['constraint_type'] == method]
                
                if len(df_subset) > 0:
                    mse_mean = df_subset['fitting_error_mse'].mean()
                    mse_std = df_subset['fitting_error_mse'].std()
                    cost_mean = df_subset['deployment_cost_l1'].mean()
                    cost_std = df_subset['deployment_cost_l1'].std()
                    unreal_mean = df_subset['unrealizable_ratio'].mean()
                    unreal_std = df_subset['unrealizable_ratio'].std()
                    
                    method_name = 'CritiPro' if method == 'ours' else 'Random'
                    print(f"  {method_name}:")
                    print(f"    Fitting Error (MSE): {mse_mean:.2f} ± {mse_std:.2f}")
                    print(f"    Deployment Cost (L1): {cost_mean:.2f} ± {cost_std:.2f}")
                    print(f"    Unrealizable Ratio: {unreal_mean:.2f}% ± {unreal_std:.2f}%")
            
            # 计算改进
            random_mse = df_topo[df_topo['constraint_type'] == 'random']['fitting_error_mse'].mean()
            ours_mse = df_topo[df_topo['constraint_type'] == 'ours']['fitting_error_mse'].mean()
            
            random_unreal = df_topo[df_topo['constraint_type'] == 'random']['unrealizable_ratio'].mean()
            ours_unreal = df_topo[df_topo['constraint_type'] == 'ours']['unrealizable_ratio'].mean()
            
            if random_mse > 0:
                mse_improvement = (random_mse - ours_mse) / random_mse * 100
                print(f"  → MSE Reduction: {mse_improvement:.1f}%")
            
            if random_unreal > 0:
                unreal_improvement = (random_unreal - ours_unreal) / random_unreal * 100
                print(f"  → Unrealizable Ratio Reduction: {unreal_improvement:.1f}%")
    
    def generate_latex_table(self, filename='comparison_table.tex'):
        """
        生成 LaTeX 表格
        """
        grouped = self.df.groupby(['topo_num', 'constraint_type']).agg({
            'fitting_error_mse': ['mean', 'std'],
            'deployment_cost_l1': ['mean', 'std'],
            'unrealizable_ratio': ['mean', 'std']
        }).round(2)
        
        latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Comparison: Random vs. CritiPro}
\label{tab:comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Topology} & \textbf{Method} & \textbf{MSE} & \textbf{Cost (L1)} & \textbf{Unreal. (\%)} \\
\midrule
"""
        
        for topo in sorted(grouped.index.get_level_values(0).unique()):
            latex_code += f"\\multirow{{2}}{{*}}{{{topo.replace('topo_', 'Topo ')}}}\n"
            
            for method in ['random', 'ours']:
                method_name = 'CritiPro' if method == 'ours' else 'Random'
                
                try:
                    mse_mean = grouped.loc[(topo, method), ('fitting_error_mse', 'mean')]
                    mse_std = grouped.loc[(topo, method), ('fitting_error_mse', 'std')]
                    cost_mean = grouped.loc[(topo, method), ('deployment_cost_l1', 'mean')]
                    cost_std = grouped.loc[(topo, method), ('deployment_cost_l1', 'std')]
                    unreal_mean = grouped.loc[(topo, method), ('unrealizable_ratio', 'mean')]
                    unreal_std = grouped.loc[(topo, method), ('unrealizable_ratio', 'std')]
                    
                    latex_code += f"    & {method_name} & ${mse_mean:.2f} \\pm {mse_std:.2f}$ & "
                    latex_code += f"${cost_mean:.2f} \\pm {cost_std:.2f}$ & "
                    latex_code += f"${unreal_mean:.2f} \\pm {unreal_std:.2f}$ \\\\\n"
                except KeyError:
                    latex_code += f"    & {method_name} & N/A & N/A & N/A \\\\\n"
            
            latex_code += "\\midrule\n"
        
        latex_code += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(latex_code)
        
        print(f"✓ LaTeX 表格已保存: {output_path}")


def main():
    """
    主函数：生成所有图表
    """
    # 使用你的结果文件路径
    results_path = '/home/retr0/Project/TopologyObfu/Experiment/ablation/comparison_results/comparison_results.json'
    
    # 如果文件不存在，尝试其他可能的路径
    if not os.path.exists(results_path):
        alternative_paths = [
            '/home/retr0/Project/TopologyObfu/Experiment/comparison/comparison_results/comparison_results.json',
            './ablation_results/comparison_results.json'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                results_path = path
                break
    
    # 加载结果
    plotter = ComparisonPlotter(results_path)
    
    # 打印统计摘要
    plotter.print_summary_statistics()
    
    # 生成图表
    print("\n" + "="*60)
    print("Generating High-Resolution Plots...")
    print("="*60 + "\n")
    
    # 1. MSE 对比图（独立）
    plotter.plot_mse_comparison(
        filename='comparison_mse.png',
        figsize=(10, 6),
        dpi=300
    )
    
    # 2. Unrealizable Ratio 对比图（独立）
    plotter.plot_unrealizable_comparison(
        filename='comparison_unrealizable.png',
        figsize=(10, 6),
        dpi=300
    )
    
    # 3. Deployment Cost 对比图
    plotter.plot_deployment_cost(
        filename='comparison_deployment_cost.png',
        figsize=(10, 6),
        dpi=300
    )
    
    # 4. 生成 LaTeX 表格
    plotter.generate_latex_table(
        filename='comparison_table.tex'
    )
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
