使用说明文档

<MARKDOWN>
# 暴力故障影响分析实验
## 实验目标
验证关键节点识别算法（基于 TOPSIS）的有效性，通过遍历所有交换机节点，模拟其故障，测量对网络吞吐量的影响，与算法识别结果进行对比。
---
## 文件说明
| 文件名 | 说明 |
|--------|------|
| `brute_force_failure_analysis.py` | 主实验脚本（暴力遍历 + TOPSIS 整合） |
| `visualize_results.py` | 可视化脚本（散点图、柱状图、热力图） |
| `my_learning_switch.py` | Ryu 控制器程序（需提前启动） |
| `README.md` | 本文档 |
---
## 实验流程
### 步骤 1: 准备工作
**1.1 确保拓扑文件存在**
/home/retr0/Project/TopologyObfu/CritiPro/topo_X_result/
├── topo_X.txt          # 邻接矩阵
├── topo_X_info.txt     # 主机连接信息
└── metrics.txt         # 节点指标（TOPSIS 输入）

<TEXT>
**1.2 启动 Ryu 控制器**
```bash
# 在终端 1 中运行
cd /home/retr0/Project/TopologyObfu/
ryu-manager --verbose --ofp-tcp-listen-port 6633 my_learning_switch_stpban.py
1.3 清理 Mininet 环境（可选）

<BASH>
sudo mn -c
步骤 2: 运行主实验
在终端 2 中运行：

<BASH>
sudo ~/anaconda3/bin/python brute_force_failure_analysis.py
交互式输入：

<TEXT>
请输入拓扑编号（例如 1）: 1
实验过程：

<TEXT>
[INFO] 阶段 1: 基准测试
       ↓
       测量正常网络的总吞吐量
       
[INFO] 阶段 2: 暴力故障遍历
       ↓
       For each switch:
         - 下线所有链路
         - 测量吞吐量
         - 计算影响分数
         - 恢复链路
       
[INFO] 阶段 3: 整合 TOPSIS 结果
       ↓
       运行关键节点识别算法
       合并结果并保存 CSV
预计耗时：

10 节点拓扑：~5 分钟
50 节点拓扑：~20 分钟
200 节点拓扑：~70 分钟
步骤 3: 可视化分析
<BASH>
python visualize_results.py 1
输出文件：

<TEXT>
brute_force_results/
├── raw_results_1.json                  # 原始实验数据
├── integrated_results_1.csv            # 整合后的 CSV
├── correlation_scatter_1.png           # 相关性散点图 ⭐
├── top_10_comparison_1.png             # Top-10 对比柱状图 ⭐
├── heatmap_1.png                       # 分布热力图
└── analysis_report_1.txt               # 文本分析报告 ⭐