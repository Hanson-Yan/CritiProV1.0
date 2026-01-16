import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D  # 添加这个导入


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

def plot_model_values_curve(detection_counts, file_prefix=""):
    """
    绘制三种模型的取值折线图。
    
    参数:
        detection_counts (list): 探测次数列表，例如 [500, 1000, 2000, 3000, 5000, 7000, 10000]
        file_prefix (str): 文件名前缀，默认为空字符串。如果文件名有前缀，可以在这里指定。
    """
    # 初始化存储三种模型取值的列表
    critipro_values = []
    proto_values = []
    antitomo_values = []

    # 遍历文件名读取数据
    for count in detection_counts:
        # 构造文件名
        filename = f"{file_prefix}{count}.txt"
        
        try:
            # 打开文件并读取数据
            with open(filename, 'r') as file:
                content = file.read().strip()  # 读取文件内容并去除多余空格和换行符
                data = parse_file_content(content)  # 解析文件内容
                
                # 提取每种模型的第二个值
                critipro_values.append(data['critipro'][1])
                proto_values.append(data['proto'][1])
                antitomo_values.append(data['antitomo'][1])
        except FileNotFoundError:
            print(f"文件 {filename} 未找到，请检查文件路径和文件名是否正确。")
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误：{e}")

    return critipro_values, proto_values, antitomo_values

# ==================== 字体大小配置 ====================
FONT_CONFIG = {
    'suptitle': 18,      # 总标题
    'title': 16,         # 子图标题
    'xlabel': 14,        # x轴标签
    'ylabel': 14,        # y轴标签
    'ticks': 12,         # 坐标轴刻度
    'legend': 12,        # 图例
    'markersize': 6      # 数据点标记大小
}
# ====================================================
# 主程序
detection_counts = [500, 1000, 2000, 3000, 5000, 7000, 10000]
topo_nums = [1, 2, 3, 4]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# fig.suptitle('Delay Impact for Different Topologies', fontsize=FONT_CONFIG['suptitle'])
for i, num in enumerate(topo_nums):
    file_prefix = f"/home/retr0/Project/TopologyObfu/Experiment/topo_{num}_result/deploy_cost/"
    critipro_values, proto_values, antitomo_values = plot_model_values_curve(detection_counts, file_prefix)
    if i == 2:
        ax = axs[1, 1]
    elif i == 3:
        ax = axs[1, 0]
    else:
        ax = axs[i // 2, i % 2]
    
    x = np.array(detection_counts)
    x_new = np.linspace(x.min(), x.max(), 300)
    critipro_spline = interp1d(x, critipro_values, kind='cubic', fill_value="extrapolate")
    proto_spline = interp1d(x, proto_values, kind='cubic', fill_value="extrapolate")
    antitomo_spline = interp1d(x, antitomo_values, kind='cubic', fill_value="extrapolate")
    ax.plot(x_new, critipro_spline(x_new), label='CritiPro', linestyle='-', linewidth=2, color='C0')
    ax.plot(x, critipro_values, marker='o', linestyle='None', color='C0', markersize=FONT_CONFIG['markersize'])
    ax.plot(x_new, proto_spline(x_new), label='ProTO', linestyle='-', linewidth=2, color='C1')
    ax.plot(x, proto_values, marker='s', linestyle='None', color='C1', markersize=FONT_CONFIG['markersize'])
    ax.plot(x_new, antitomo_spline(x_new), label='AntiTomo', linestyle='-', linewidth=2, color='C2')
    ax.plot(x, antitomo_values, marker='^', linestyle='None', color='C2', markersize=FONT_CONFIG['markersize'])
    if num==3:
        num=4
    elif num==4:
        num=3
    ax.set_title(f'Topology {num}', fontsize=FONT_CONFIG['title'])
    ax.set_xlabel('Detection frequency', fontsize=FONT_CONFIG['xlabel'])
    ax.set_ylabel('Delay overhead', fontsize=FONT_CONFIG['ylabel'])
    ax.tick_params(axis='both', labelsize=FONT_CONFIG['ticks'])
    
    legend_elements = [
        Line2D([0], [0], color='C0', marker='o', linestyle='-', linewidth=2, markersize=FONT_CONFIG['markersize'], label='CritiPro'),
        Line2D([0], [0], color='C1', marker='s', linestyle='-', linewidth=2, markersize=FONT_CONFIG['markersize'], label='ProTO'),
        Line2D([0], [0], color='C2', marker='^', linestyle='-', linewidth=2, markersize=FONT_CONFIG['markersize'], label='AntiTomo')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.25), fontsize=FONT_CONFIG['legend'])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path=f"/home/retr0/Project/TopologyObfu/Experiment/all delay impact.png"
plt.savefig(save_path, format='png', dpi=600)
plt.show()