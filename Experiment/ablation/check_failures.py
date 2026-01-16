import json
from collections import defaultdict

# 读取结果
with open('/home/retr0/Project/TopologyObfu/Experiment/ablation/ablation_results/ablation_results.json', 'r') as f:
    results = json.load(f)

# 分析失败原因
failed = [r for r in results if r['status'] == 'failed']

print(f"总实验数: {len(results)}")
print(f"成功: {len([r for r in results if r['status'] == 'success'])}")
print(f"失败: {len(failed)}\n")

# 按错误类型分组
error_types = defaultdict(list)
for exp in failed:
    error_msg = exp['error_message']
    # 提取错误类型（取第一行）
    error_type = error_msg.split('\n')[0] if error_msg else "Unknown"
    error_types[error_type].append(
        f"{exp['topo_num']}_prob{exp['prob_num']}_{exp['constraint_type']}"
    )

print("="*60)
print("失败原因统计:")
print("="*60)
for error_type, exps in error_types.items():
    print(f"\n错误类型: {error_type}")
    print(f"数量: {len(exps)}")
    print(f"实验: {exps[:5]}")  # 只显示前5个
    if len(exps) > 5:
        print(f"  ... 还有 {len(exps)-5} 个")

# 按拓扑统计
print("\n" + "="*60)
print("各拓扑失败统计:")
print("="*60)
for topo in ["topo_1", "topo_2", "topo_3", "topo_4"]:
    topo_failed = [e for e in failed if e['topo_num'] == topo]
    topo_total = len([r for r in results if r['topo_num'] == topo])
    print(f"{topo}: {len(topo_failed)}/{topo_total} 失败")
    
    # 显示失败的探测数量
    if topo_failed:
        failed_probs = set(e['prob_num'] for e in topo_failed)
        print(f"  失败的探测数量: {sorted(failed_probs)}")
