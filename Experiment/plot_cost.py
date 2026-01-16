# import matplotlib.pyplot as plt
# import re
# import numpy as np
# from scipy.interpolate import interp1d

# def parse_file_content(content):
#     """
#     解析文件内容，提取模型的值。
    
#     参数:
#         content (str): 文件内容字符串。
    
#     返回:
#         dict: 包含模型名称和对应值的字典。
#     """
#     # 使用正则表达式匹配模型名称和值
#     pattern = re.compile(r"(\w+): \((\d+),\s*(\d+\.\d+)\)")
#     matches = pattern.findall(content)
    
#     # 将匹配结果转换为字典
#     result = {model: (int(value1), float(value2)) for model, value1, value2 in matches}
#     return result

# def plot_model_values_curve(detection_counts, file_prefix=""):
#     """
#     绘制三种模型的取值折线图。
    
#     参数:
#         detection_counts (list): 探测次数列表，例如 [500, 1000, 2000, 3000, 5000, 7000, 10000]
#         file_prefix (str): 文件名前缀，默认为空字符串。如果文件名有前缀，可以在这里指定。
#     """
#     # 初始化存储三种模型取值的列表
#     critipro_values = []
#     proto_values = []
#     antitomo_values = []

#     # 遍历文件名读取数据
#     for count in detection_counts:
#         # 构造文件名
#         filename = f"{file_prefix}{count}.txt"
        
#         try:
#             # 打开文件并读取数据
#             with open(filename, 'r') as file:
#                 # data = eval(file.read())  # 将文件内容转换为字典
#                 content = file.read().strip()  # 读取文件内容并去除多余空格和换行符
#                 data = parse_file_content(content)  # 解析文件内容
                
#                 # 提取每种模型的第一个值
#                 critipro_values.append(data['critipro'][0])
#                 proto_values.append(data['proto'][0])
#                 antitomo_values.append(data['antitomo'][0])
#         except FileNotFoundError:
#             print(f"文件 {filename} 未找到，请检查文件路径和文件名是否正确。")
#         except Exception as e:
#             print(f"读取文件 {filename} 时发生错误：{e}")

#     # 将横坐标改为均匀分布
#     # x = np.arange(len(detection_counts))

#     # # 使用插值方法平滑曲线
#     # x_new = np.linspace(0, len(detection_counts) - 1, 300)  # 生成更密集的横坐标点
#     # critipro_smooth = np.interp(x_new, x, critipro_values)
#     # proto_smooth = np.interp(x_new, x, proto_values)
#     # antitomo_smooth = np.interp(x_new, x, antitomo_values)

#     # # 添加轻微的随机扰动，但减少扰动幅度
#     # np.random.seed(42)  # 设置随机种子以确保结果可复现
#     # jitter = 0.5  # 扰动幅度
#     # critipro_jittered = critipro_smooth + np.random.uniform(-jitter, jitter, len(x_new))
#     # proto_jittered = proto_smooth + np.random.uniform(-jitter, jitter, len(x_new))
#     # antitomo_jittered = antitomo_smooth + np.random.uniform(-jitter, jitter, len(x_new))

#     # # 绘制折线图
#     # plt.figure(figsize=(10, 6))  # 设置图像大小
#     # plt.plot(x_new, critipro_jittered, label='Critipro', linewidth=2)
#     # plt.plot(x_new, proto_jittered, label='Proto', linewidth=2)
#     # plt.plot(x_new, antitomo_jittered, label='Antitomo', linewidth=2)

#     # # 添加标题和标签
#     # plt.title('Model Values vs Detection Counts')
#     # plt.xlabel('Detection Counts')
#     # plt.ylabel('Model Values')
#     # plt.legend()  # 添加图例

#     # # 设置横坐标刻度标签
#     # plt.xticks(x, detection_counts, rotation=45)  # 使用原始探测次数作为刻度标签

#     # # 不显示网格线
#     # plt.grid(False)

#     # # 显示图像
#     # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
#     # plt.show()
#      # 将横坐标改为均匀分布
#     x = np.arange(len(detection_counts))

#     # 绘制折线图
#     plt.figure(figsize=(10, 6))  # 设置图像大小
#     plt.plot(x, critipro_values, marker='o', label='Critipro', linestyle='-', linewidth=2)
#     plt.plot(x, proto_values, marker='s', label='Proto', linestyle='-', linewidth=2)
#     plt.plot(x, antitomo_values, marker='^', label='Antitomo', linestyle='-', linewidth=2)

#     # 添加标题和标签
#     plt.title('Model Values vs Detection Counts')
#     plt.xlabel('Detection Counts')
#     plt.ylabel('Model Values')
#     plt.legend()  # 添加图例

#     # 设置横坐标刻度标签
#     plt.xticks(x, detection_counts, rotation=45)  # 使用原始探测次数作为刻度标签

#     # 不显示网格线
#     plt.grid(False)

#     # 显示图像
#     plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
#     plt.show()

# def plot_model_values_pillar(detection_counts, file_prefix=""):
#     """
#     绘制三种模型的取值柱状图。
    
#     参数:
#         detection_counts (list): 探测次数列表，例如 [500, 1000, 2000, 3000, 5000, 7000, 10000]
#         file_prefix (str): 文件名前缀，默认为空字符串。如果文件名有前缀，可以在这里指定。
#     """
#     # 初始化存储三种模型取值的列表
#     critipro_values = []
#     proto_values = []
#     antitomo_values = []

#     # 遍历文件名读取数据
#     for count in detection_counts:
#         # 构造文件名
#         filename = f"{file_prefix}{count}.txt"
        
#         try:
#             # 打开文件并读取数据
#             with open(filename, 'r') as file:
#                 content = file.read().strip()  # 读取文件内容并去除多余空格和换行符
#                 data = parse_file_content(content)  # 解析文件内容
                
#                 # 提取每种模型的第一个值
#                 critipro_values.append(data['critipro'][0])
#                 proto_values.append(data['proto'][0])
#                 antitomo_values.append(data['antitomo'][0])
#         except FileNotFoundError:
#             print(f"文件 {filename} 未找到，请检查文件路径和文件名是否正确。")
#         except Exception as e:
#             print(f"读取文件 {filename} 时发生错误：{e}")

#     # 设置柱状图的位置和宽度
#     n_groups = len(detection_counts)
#     bar_width = 0.2
#     index = np.arange(n_groups)

#     # 定义浅色调的颜色
#     # colors = ['#E7EFFA', '#F7E1ED', '#A0EEE1']  # 浅蓝、浅粉、浅绿
#     colors = ['#63D2F9', '#EB624F', '#90D26C']
#     # colors = ["#0072B2", "#E69F00", "#009E73"]  # 浅蓝、浅粉、浅绿
    
#     # 定义条纹纹理
#     # hatches = ['//', '..', '**']  # 不同的条纹纹理

#     # 绘制柱状图
#     plt.figure(figsize=(12, 6))  # 设置图像大小
#     # bars1 = plt.bar(index, critipro_values, bar_width, label='Critipro', alpha=0.8, color=colors[0], hatch=hatches[0])
#     # bars2 = plt.bar(index + bar_width, proto_values, bar_width, label='Proto', alpha=0.8, color=colors[1], hatch=hatches[1])
#     # bars3 = plt.bar(index + 2 * bar_width, antitomo_values, bar_width, label='Antitomo', alpha=0.8, color=colors[2], hatch=hatches[2])
#     bars1 = plt.bar(index, critipro_values, bar_width, label='Critipro', alpha=0.8, color=colors[0])
#     bars2 = plt.bar(index + bar_width, proto_values, bar_width, label='Proto', alpha=0.8, color=colors[1])
#     bars3 = plt.bar(index + 2 * bar_width, antitomo_values, bar_width, label='Antitomo', alpha=0.8, color=colors[2])

#     # 添加标题和标签
#     plt.title('Topology')
#     plt.xlabel('Detection frequency')
#     plt.ylabel('Deployment Cost')
#     plt.legend()  # 添加图例

#     # 设置横坐标刻度标签
#     plt.xticks(index + bar_width, detection_counts, rotation=45)  # 使用原始探测次数作为刻度标签

#     # 不显示网格线
#     plt.grid(False)

#     # 显示图像
#     plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
#     plt.show()




# # 示例调用
# detection_counts = [500, 1000, 2000, 3000, 5000, 7000, 10000]
# topo_num=input("please input topo_num:")
# file_prefix=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/deploy_cost/"
# plot_model_values_pillar(detection_counts,file_prefix)
import matplotlib.pyplot as plt
import re
import numpy as np

def parse_file_content(content):
    """
    解析文件内容，提取模型的值。
    
    参数:
        content (str): 文件内容字符串。
    
    返回:
        dict: 包含模型名称和对应值的字典。
    """
    # 使用正则表达式匹配模型名称和值
    pattern = re.compile(r"(\w+): \((\d+),\s*(\d+\.\d+)\)")
    matches = pattern.findall(content)
    
    # 将匹配结果转换为字典
    result = {model: (int(value1), float(value2)) for model, value1, value2 in matches}
    return result

# def plot_model_values_pillar(detection_counts, file_prefix=""):
#     """
#     绘制三种模型的取值柱状图。
    
#     参数:
#         detection_counts (list): 探测次数列表，例如 [500, 1000, 2000, 3000, 5000, 7000, 10000]
#         file_prefix (str): 文件名前缀，默认为空字符串。如果文件名有前缀，可以在这里指定。
#     """
#     # 初始化存储三种模型取值的列表
#     critipro_values = []
#     proto_values = []
#     antitomo_values = []

#     # 遍历文件名读取数据
#     for count in detection_counts:
#         # 构造文件名
#         filename = f"{file_prefix}{count}.txt"
        
#         try:
#             # 打开文件并读取数据
#             with open(filename, 'r') as file:
#                 content = file.read().strip()  # 读取文件内容并去除多余空格和换行符
#                 data = parse_file_content(content)  # 解析文件内容
                
#                 # 提取每种模型的第一个值
#                 critipro_values.append(data['critipro'][0])
#                 proto_values.append(data['proto'][0])
#                 antitomo_values.append(data['antitomo'][0])
#         except FileNotFoundError:
#             print(f"文件 {filename} 未找到，请检查文件路径和文件名是否正确。")
#         except Exception as e:
#             print(f"读取文件 {filename} 时发生错误：{e}")

#     # 设置柱状图的位置和宽度
#     n_groups = len(detection_counts)
#     bar_width = 0.2
#     index = np.arange(n_groups)

#     # 定义颜色
#     colors = ['#63D2F9', '#EB624F', '#90D26C']
    
#     # 绘制柱状图
#     topo_num = file_prefix.split('/')[-2].split('_')[0]  # 提取拓扑编号
#     plt.title(f'Topology {topo_num}')
#     plt.xlabel('Detection frequency')
#     plt.ylabel('Deployment Cost')
#     plt.legend()  # 添加图例

#     # 设置横坐标刻度标签
#     plt.xticks(index + bar_width, detection_counts, rotation=45)  # 使用原始探测次数作为刻度标签

#     # 不显示网格线
#     plt.grid(False)

#     # 自动调整子图参数，使之填充整个图像区域
#     plt.tight_layout()

#     return critipro_values, proto_values, antitomo_values
def plot_model_values_pillar(detection_counts, file_prefix=""):
    """
    读取三种模型的取值数据。
    
    参数:
        detection_counts (list): 探测次数列表
        file_prefix (str): 文件名前缀
    
    返回:
        tuple: (critipro_values, proto_values, antitomo_values)
    """
    # 初始化存储三种模型取值的列表
    critipro_values = []
    proto_values = []
    antitomo_values = []
    # 遍历文件名读取数据
    for count in detection_counts:
        filename = f"{file_prefix}{count}.txt"
        
        try:
            with open(filename, 'r') as file:
                content = file.read().strip()
                data = parse_file_content(content)
                
                critipro_values.append(data['critipro'][0])
                proto_values.append(data['proto'][0])
                antitomo_values.append(data['antitomo'][0])
        except FileNotFoundError:
            print(f"文件 {filename} 未找到,请检查文件路径和文件名是否正确。")
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误:{{e}}")
    return critipro_values, proto_values, antitomo_values
# ==================== 字体大小配置 ====================
FONT_CONFIG = {
    'suptitle': 18,      # 总标题
    'title': 16,         # 子图标题
    'xlabel': 14,        # x轴标签
    'ylabel': 14,        # y轴标签
    'xticks': 12,        # x轴刻度
    'yticks': 12,        # y轴刻度
    'legend': 12         # 图例
}
# ====================================================

# 主程序
detection_counts = [500, 1000, 2000, 3000, 5000, 7000, 10000]
topo_nums = [1, 2, 4, 3]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# fig.suptitle('Deployment Cost for Different Topologies', fontsize=FONT_CONFIG['suptitle'])

for i, num in enumerate(topo_nums):
    file_prefix = f"/home/retr0/Project/TopologyObfu/Experiment/topo_{num}_result/deploy_cost/"
    critipro_values, proto_values, antitomo_values = plot_model_values_pillar(detection_counts, file_prefix)
    
    ax = axs[i // 2, i % 2]
    
    n_groups = len(detection_counts)
    bar_width = 0.2
    index = np.arange(n_groups)
    colors = ['#63D2F9', '#EB624F', '#90D26C']
    
    ax.bar(index, critipro_values, bar_width, label='CritiPro', alpha=0.8, color=colors[0])
    ax.bar(index + bar_width, proto_values, bar_width, label='ProTO', alpha=0.8, color=colors[1])
    ax.bar(index + 2 * bar_width, antitomo_values, bar_width, label='AntiTomo', alpha=0.8, color=colors[2])
    
    if num==3:
        num=4
    elif num==4:
        num=3
    ax.set_title(f'Topology {num}', fontsize=FONT_CONFIG['title'])
    ax.set_xlabel('Detection frequency', fontsize=FONT_CONFIG['xlabel'])
    ax.set_ylabel('Deployment overhead', fontsize=FONT_CONFIG['ylabel'])
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(detection_counts, rotation=45, fontsize=FONT_CONFIG['xticks'])
    ax.tick_params(axis='y', labelsize=FONT_CONFIG['yticks'])
    ax.legend(fontsize=FONT_CONFIG['legend'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path=f"/home/retr0/Project/TopologyObfu/Experiment/all cost statistic.png"
plt.savefig(save_path, format='png', dpi=600)
plt.show()
