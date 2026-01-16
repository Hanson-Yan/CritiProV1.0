# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # 指定新的颜色列表
# colors = ['#63D2F9', '#EB624F', '#90D26C']

# def read_all_json_files():
#     """读取指定目录下所有JSON文件的内容，返回包含所有JSON内容的列表"""
#     json_list = []
#     current_dir = "/home/retr0/Project/TopologyObfu/Experiment/flood_test/experiment1/"
#     print(f"正在从目录读取JSON文件: {current_dir}")
    
#     # 遍历目录下的所有JSON文件
#     json_files_found = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    
#     if not json_files_found:
#         print("警告: 在当前目录下未找到任何.json文件")
#         return json_list
    
#     print(f"找到{len(json_files_found)}个JSON文件，开始读取...")
    
#     for filename in json_files_found:
#         file_path = os.path.join(current_dir, filename)
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 json_content = file.read()
#                 # 验证JSON格式
#                 json.loads(json_content)
#                 json_list.append(json_content)
#                 print(f"成功读取: {filename}")
                
#         except json.JSONDecodeError:
#             print(f"跳过: {filename} 不是有效的JSON格式")
#         except Exception as e:
#             print(f"读取{filename}失败: {str(e)}")
    
#     return json_list

# # 读取JSON文件
# json_list = read_all_json_files()

# # 交换索引2和3的元素（从0开始）
# if len(json_list) >= 4:
#     # 交换元素
#     json_list[2], json_list[3] = json_list[3], json_list[2]
#     print("已成功交换索引2和3的元素")
# else:
#     print("无法交换元素：列表包含的元素数量不足4个")

# # 检查是否读取到有效文件
# if not json_list:
#     print("错误: 没有读取到任何有效的JSON文件，程序无法继续执行")
#     exit(1)

# # 解析JSON数据
# try:
#     datasets = [json.loads(j) for j in json_list]
# except json.JSONDecodeError as e:
#     print(f"解析JSON数据时出错: {e}")
#     exit(1)

# # 检查数据集是否有效
# if not datasets:
#     print("错误: 数据集为空，无法生成图表")
#     exit(1)

# num_topologies = len(datasets)
# scenarios = ["baseline", "true_attack", "fake_attack"]  # 明确指定场景顺序
# scenario_labels = ["Baseline", "Real Attack", "Obfuscated Attack"]  # 场景显示名称

# # 准备数据矩阵
# try:
#     throughput_matrix = np.array([
#         [d[sc]['throughput_Mbps'] for sc in scenarios] for d in datasets
#     ])

#     loss_matrix = np.array([
#         [d[sc]['loss_percent'] for sc in scenarios] for d in datasets
#     ])
# except KeyError as e:
#     print(f"数据中缺少必要的键: {e}")
#     exit(1)

# # X轴位置 - 每个拓扑一个主位置
# x = np.arange(num_topologies)
# bar_width = 0.25  # 每个场景的条形宽度

# # 吞吐量分组柱状图
# plt.figure(figsize=(10, 6))
# for i in range(len(scenarios)):
#     # 每个场景在拓扑位置上的偏移，并使用指定颜色
#     plt.bar(x + i * bar_width, throughput_matrix[:, i], width=bar_width, 
#             label=scenario_labels[i], color=colors[i])

# plt.xticks(x + bar_width, [f"Topology {i+1}" for i in range(num_topologies)])
# plt.ylabel('Throughput (Mbps)')
# plt.title('Throughput Comparison Across Topologies')
# plt.legend()
# plt.tight_layout()
# save_path=f"/home/retr0/Project/TopologyObfu/Experiment/flood_test/experiment1/flood_throughout.png"
# plt.savefig(save_path, format='png', dpi=600)
# plt.show()

# # 丢包率分组柱状图
# plt.figure(figsize=(10, 6))
# for i in range(len(scenarios)):
#     # 使用指定颜色
#     plt.bar(x + i * bar_width, loss_matrix[:, i] * 100, width=bar_width, 
#             label=scenario_labels[i], color=colors[i])

# plt.xticks(x + bar_width, [f"Topology {i+1}" for i in range(num_topologies)])
# plt.ylabel('Packet Loss (%)')
# plt.title('Packet Loss Comparison Across Topologies')
# plt.legend()
# plt.tight_layout()
# save_path=f"/home/retr0/Project/TopologyObfu/Experiment/flood_test/experiment1/flood_loss.png"
# plt.savefig(save_path, format='png', dpi=600)
# plt.show()
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# 指定新的颜色列表
colors = ['#63D2F9', '#EB624F', '#90D26C']

plt.rcParams.update({
    # 'font.size': 14,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    # 'figure.titlesize': 15,
})

def read_all_json_files():
    """读取指定目录下所有JSON文件的内容，返回包含所有JSON内容的列表
    列表顺序与文件名中的数字对应：results_topo_1.json -> 索引0，results_topo_2.json -> 索引1
    """
    json_list = []
    current_dir = "/home/retr0/Project/TopologyObfu/Experiment/flood_test/experiment1/"
    print(f"正在从目录读取JSON文件: {current_dir}")
    
    # 遍历目录下的所有JSON文件
    json_files_found = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    
    if not json_files_found:
        print("警告: 在当前目录下未找到任何.json文件")
        return json_list
    
    # 定义提取数字的函数
    def extract_topo_number(filename):
        """从文件名中提取topo的数字，如results_topo_5.json返回5"""
        # 使用正则表达式匹配数字部分
        match = re.search(r'results_topo_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        # 如果文件名不符合格式，返回一个很大的数放到最后
        return float('inf')
    
    # 按提取的数字排序文件列表
    json_files_found.sort(key=extract_topo_number)
    
    print(f"找到{len(json_files_found)}个JSON文件，按数字排序后开始读取...")
    print(f"文件顺序: {json_files_found}")
    
    for filename in json_files_found:
        file_path = os.path.join(current_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_content = file.read()
                # 验证JSON格式
                json.loads(json_content)
                json_list.append(json_content)
                print(f"成功读取: {filename}")
                
        except json.JSONDecodeError:
            print(f"跳过: {filename} 不是有效的JSON格式")
        except Exception as e:
            print(f"读取{filename}失败: {str(e)}")
    
    return json_list

# 读取JSON文件
json_list = read_all_json_files()

# 交换索引2和3的元素（从0开始）
if len(json_list) >= 4:
    # 交换元素
    json_list[2], json_list[3] = json_list[3], json_list[2]
    print("已成功交换索引2和3的元素")
else:
    print("无法交换元素：列表包含的元素数量不足4个")

# 检查是否读取到有效文件
if not json_list:
    print("错误: 没有读取到任何有效的JSON文件，程序无法继续执行")
    exit(1)

# 解析JSON数据
try:
    datasets = [json.loads(j) for j in json_list]
except json.JSONDecodeError as e:
    print(f"解析JSON数据时出错: {e}")
    exit(1)

# 检查数据集是否有效
if not datasets:
    print("错误: 数据集为空，无法生成图表")
    exit(1)

num_topologies = len(datasets)
scenarios = ["baseline", "true_attack", "fake_attack"]  # 明确指定场景顺序
scenario_labels = ["Baseline", "Real Attack", "Obfuscated Attack"]  # 场景显示名称

# 准备数据矩阵
try:
    throughput_matrix = np.array([
        [d[sc]['throughput_Mbps'] for sc in scenarios] for d in datasets
    ])

    loss_matrix = np.array([
        [d[sc]['loss_percent'] for sc in scenarios] for d in datasets
    ])
except KeyError as e:
    print(f"数据中缺少必要的键: {e}")
    exit(1)

# X轴位置 - 每个拓扑一个主位置
x = np.arange(num_topologies)
bar_width = 0.25  # 每个场景的条形宽度

# 创建一个包含两个子图的图形（1行2列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))  # 增大宽度以容纳两个图

# 吞吐量分组柱状图（左图）
for i in range(len(scenarios)):
    ax1.bar(x + i * bar_width, throughput_matrix[:, i], width=bar_width, 
            label=scenario_labels[i], color=colors[i])

ax1.set_xticks(x + bar_width)
ax1.set_xticklabels([f"Topology {i+1}" for i in range(num_topologies)])
ax1.set_ylabel('Throughput (Mbps)')
ax1.set_title('Throughput Comparison Across Topologies')
ax1.legend()

# 丢包率分组柱状图（右图）
for i in range(len(scenarios)):
    ax2.bar(x + i * bar_width, loss_matrix[:, i] * 100, width=bar_width, 
            label=scenario_labels[i], color=colors[i])

ax2.set_xticks(x + bar_width)
ax2.set_xticklabels([f"Topology {i+1}" for i in range(num_topologies)])
ax2.set_ylabel('Packet Loss (%)')
ax2.set_title('Packet Loss Comparison Across Topologies')
ax2.legend()

# 调整子图之间的间距
plt.tight_layout()

# 保存组合图
save_path = f"/home/retr0/Project/TopologyObfu/Experiment/flood_test/experiment1/flood_comparison.pdf"
plt.savefig(save_path, format='pdf')
plt.show()
