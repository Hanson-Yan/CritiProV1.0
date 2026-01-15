from draw_topo import DrawTopology
import numpy as np


original_matrix = np.loadtxt("/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation/data/input_adj.txt")
final_matrix = np.loadtxt("/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation/data/output_adj.txt")

DrawTopology(matrix=original_matrix,critical_nodes=None).draw(show=True)
DrawTopology(matrix=final_matrix,critical_nodes=None).draw(show=True,save_path="data/confuse_topo.png")