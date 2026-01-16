import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

# =========================================================
# ✅ 字体样式配置（统一控制）
# =========================================================
FONT_SIZES = {
    'title': 18,          # 总标题
    'subtitle': 20,       # 每个子图标题
    'label': 19,          # 坐标轴标签
    'ticks': 12,          # 坐标轴刻度
    'legend': 15          # 图例文字
}

# 应用到 matplotlib 全局设置
plt.rcParams.update({
    'font.size': FONT_SIZES['ticks'],   # 默认字体
    'axes.titlesize': FONT_SIZES['subtitle'],
    'axes.labelsize': FONT_SIZES['label'],
    'legend.fontsize': FONT_SIZES['legend'],
    'xtick.labelsize': FONT_SIZES['ticks'],
    'ytick.labelsize': FONT_SIZES['ticks'],
    'figure.titlesize': FONT_SIZES['title']
})
# =========================================================


def parse_file_content(content):
    """解析文件内容，提取模型的值。"""
    pattern = re.compile(r"(\w+): \((\d+),\s*(\d+\.\d+)\)")
    matches = pattern.findall(content)
    result = {model: (int(value1), float(value2)) for model, value1, value2 in matches}
    return result


def plot_model_values_curve(detection_counts, file_prefix=""):
    """绘制三种模型的取值折线图。"""
    critipro_values, proto_values, antitomo_values = [], [], []

    for count in detection_counts:
        filename = f"{file_prefix}{count}.txt"
        try:
            with open(filename, 'r') as file:
                content = file.read().strip()
                data = parse_file_content(content)
                critipro_values.append(data['critipro'][1])
                proto_values.append(data['proto'][1])
                antitomo_values.append(data['antitomo'][1])
        except FileNotFoundError:
            print(f"文件 {filename} 未找到。")
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误：{e}")

    return critipro_values, proto_values, antitomo_values


# 主程序
detection_counts = [500, 1000, 2000, 3000, 5000, 7000, 10000]
topo_nums = [1, 2, 3, 4]

# 改为 1x4 布局
fig, axs = plt.subplots(1, 4, figsize=(20, 4))
# fig.suptitle('Delay Impact for Different Topologies', fontsize=FONT_SIZES['title'], fontweight='bold')

for i, num in enumerate(topo_nums):
    file_prefix = f"/home/retr0/Project/TopologyObfu/Experiment/topo_{num}_result/deploy_cost/"
    critipro_values, proto_values, antitomo_values = plot_model_values_curve(detection_counts, file_prefix)

    ax = axs[i]

    x = np.array(detection_counts)
    x_new = np.linspace(x.min(), x.max(), 300)

    critipro_spline = interp1d(x, critipro_values, kind='cubic', fill_value="extrapolate")
    proto_spline = interp1d(x, proto_values, kind='cubic', fill_value="extrapolate")
    antitomo_spline = interp1d(x, antitomo_values, kind='cubic', fill_value="extrapolate")

    ax.plot(x_new, critipro_spline(x_new), label='Critipro', linestyle='-', linewidth=2, color='C0')
    ax.plot(x, critipro_values, marker='o', linestyle='None', color='C0')

    ax.plot(x_new, proto_spline(x_new), label='Proto', linestyle='-', linewidth=2, color='C1')
    ax.plot(x, proto_values, marker='s', linestyle='None', color='C1')

    ax.plot(x_new, antitomo_spline(x_new), label='Antitomo', linestyle='-', linewidth=2, color='C2')
    ax.plot(x, antitomo_values, marker='^', linestyle='None', color='C2')

    # display_num = 4 if num == 3 else 3 if num == 4 else num
    ax.set_title(f'Topology {num}', fontsize=FONT_SIZES['subtitle'])
    ax.set_xlabel('Detection frequency', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Delay impact', fontsize=FONT_SIZES['label'])

    legend_elements = [
        Line2D([0], [0], color='C0', marker='o', linestyle='-', linewidth=2, label='CritiPro'),
        Line2D([0], [0], color='C1', marker='s', linestyle='-', linewidth=2, label='ProTO'),
        Line2D([0], [0], color='C2', marker='^', linestyle='-', linewidth=2, label='AntiTomo')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT_SIZES['legend'])

plt.tight_layout(rect=[0, 0.03, 1, 0.9])

# 保存为 PDF
save_path = "/home/retr0/Project/TopologyObfu/Experiment/all_delay_impact.pdf"
plt.tight_layout()
plt.savefig(save_path, format='pdf')
plt.show()
