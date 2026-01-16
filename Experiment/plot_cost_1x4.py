import matplotlib.pyplot as plt
import re
import numpy as np

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

plt.rcParams.update({
    'font.size': FONT_SIZES['ticks'],
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


def plot_model_values_pillar(detection_counts, file_prefix=""):
    """读取模型数据并返回柱状图数据。"""
    critipro_values, proto_values, antitomo_values = [], [], []

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
            print(f"文件 {filename} 未找到。")
        except Exception as e:
            print(f"读取文件 {filename} 时发生错误：{e}")

    return critipro_values, proto_values, antitomo_values


# 主程序部分
detection_counts = [500, 1000, 2000, 3000, 5000, 7000, 10000]
topo_nums = [1, 2, 3, 4]

# 改为 1×4 横向布局
fig, axs = plt.subplots(1, 4, figsize=(20, 4))
# fig.suptitle('Deployment Cost for Different Topologies', fontsize=FONT_SIZES['title'], fontweight='bold')

# 遍历拓扑编号并绘制柱状图
for i, num in enumerate(topo_nums):
    file_prefix = f"/home/retr0/Project/TopologyObfu/Experiment/topo_{num}_result/deploy_cost/"
    critipro_values, proto_values, antitomo_values = plot_model_values_pillar(detection_counts, file_prefix)

    ax = axs[i]
    n_groups = len(detection_counts)
    bar_width = 0.25
    index = np.arange(n_groups)

    colors = ['#63D2F9', '#EB624F', '#90D26C']

    # 绘制三组柱子
    ax.bar(index, critipro_values, bar_width, label='CritiPro', alpha=0.8, color=colors[0])
    ax.bar(index + bar_width, proto_values, bar_width, label='ProTO', alpha=0.8, color=colors[1])
    ax.bar(index + 2 * bar_width, antitomo_values, bar_width, label='AntiTomo', alpha=0.8, color=colors[2])

    # 修正拓扑编号顺序
    # display_num = 4 if num == 3 else 3 if num == 4 else num

    # 设置标题与坐标轴
    ax.set_title(f'Topology {num}', fontsize=FONT_SIZES['subtitle'])
    ax.set_xlabel('Detection frequency', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Deployment Cost', fontsize=FONT_SIZES['label'])

    # 设置刻度标签
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(detection_counts, rotation=45, fontsize=FONT_SIZES['ticks'])

    # 图例
    ax.legend(loc='upper left', fontsize=FONT_SIZES['legend'])

    # 去掉网格线
    ax.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.9])

# 保存为 PDF
plt.tight_layout()
save_path = "/home/retr0/Project/TopologyObfu/Experiment/all_cost_statistic.pdf"
plt.savefig(save_path, format='pdf', dpi=600)
plt.show()
