from draw_topo import DrawTopology
import numpy as np

def compare_critinode(topo_num):
    #original_topo
    original_topo=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{topo_num}.txt"
    original_criti_node=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_original.txt"
    original_png_save_path=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/original_topo.png"
    key_transefer_rate_result_path=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/key_transefer_rate.txt"
    original_topo_matrix=np.loadtxt(original_topo)
    original_criti_node_vector=np.loadtxt(original_criti_node)
    original_criti_node_vector_s=[f's{int(num)}' for num in original_criti_node_vector]
    topology = DrawTopology(original_topo_matrix, original_criti_node_vector_s)
    topology.draw(save_path=original_png_save_path,title="original")

    model_name=["critipro","proto","antitomo"]
    overlap=[]
    for name in model_name:
        #ciritipro_topo
        _topo=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/topo/{name}_{topo_num}_confuse_topo.txt"
        _criti_node=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{topo_num}_critinode_of_{name}.txt"
        _png_save_path=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/critical_node_experiment/{name}_topo.png"
        _topo_matrix=np.loadtxt(_topo)
        _criti_node_vector=np.loadtxt(_criti_node)
        _criti_node_vector_s=[f's{int(num)}' for num in _criti_node_vector]
        overlap.append(len(set(original_criti_node_vector_s) & set(_criti_node_vector_s)))
        topology = DrawTopology(_topo_matrix, _criti_node_vector_s)
        topology.draw(save_path=_png_save_path,title=name)
        # break
    key_transefer_rate=[1-(x / len(original_criti_node_vector)) for x in overlap]
    key_transefer_rate_result={key: value for key, value in zip(model_name, key_transefer_rate)}
    print(key_transefer_rate_result)
    with open(key_transefer_rate_result_path, "w") as file:
        for key, value in key_transefer_rate_result.items():
            file.write(f"{key}: {value}\n")

