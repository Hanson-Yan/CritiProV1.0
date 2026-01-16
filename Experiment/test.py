from prepare_data import prepare_topo_data,prepare_delay_data
from key_node_static import key_node_static
from compute_similarity import compute_similarity
from compute_deploy_cost import compute_deploy_cost
from compare_critinode import compare_critinode

if __name__=="__main__":

    topo_num=input("please input topo_num:")
    prepare_topo_data(topo_num)
    key_node_static(topo_num)
    compare_critinode(topo_num)
    compute_similarity(topo_num)
    prepare_delay_data(topo_num)
    compute_deploy_cost(topo_num)




    # prepare_data(topo_num)
    # key_node_static(topo_num)
    # compute_similarity()
    # count_deploy_cost()
    