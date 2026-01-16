import matplotlib.pyplot as plt
from probe_simulation import probe_simu
from compute_similarity import exe_similarity
topo_num = input("please input topo_num: ")

probe_list = [500, 1000, 2000, 3000, 5000, 7000, 10000]
similarity_scores = []

for probe in probe_list:
    probe_simu(topo_num=topo_num, probe_num=probe, bw=1000)
    similarity_result = exe_similarity(topo_num=topo_num,probe_num=probe)
    
    # 提取 similarity_score
    score = similarity_result
    similarity_scores.append(score)

print(similarity_scores)
# 绘图
plt.close('all')
plt.figure(figsize=(10, 6))
plt.plot(probe_list, similarity_scores, marker='o', linestyle='-', color='dodgerblue', linewidth=2)

plt.title("Network Tomography Simulation Curve", fontsize=14)
plt.xlabel("Probe Number", fontsize=12)
plt.ylabel("Similarity Score", fontsize=12)
plt.grid(False)
plt.ylim(0, 1.03)

# 保存图像
plot_png_path = "/home/retr0/Project/TopologyObfu/MininetTop/probe_simulation/plot_png/"
output_filename = f"{plot_png_path}{topo_num}_simulation_curve.png"
plt.savefig(output_filename)

plt.show()