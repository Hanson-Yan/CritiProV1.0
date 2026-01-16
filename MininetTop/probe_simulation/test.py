from probe_simulation import probe_simu
from compute_similarity import exe_similarity

def multi_probe():
    
    topo_num=input("please input topo_num:")
    # probe_num=int(input("please input probe_num:"))
    for probe in [500, 1000, 2000, 3000, 5000, 7000, 10000]:
        probe_simu(topo_num=topo_num,probe_num=probe,bw=1000)
        # similarity_score=exe_similarity(topo_num=topo_num)
        # topo_infer(topo_num=topo_num, probe_num=probe)
        # compute_similarity(topo_num=topo_num)

def main():
    topo_num=input("please input topo_num:")
    probe_num=int(input("please input probe_num:"))
    probe_simu(topo_num=topo_num,probe_num=probe_num,bw=1000)
    exe_similarity(topo_num=topo_num)
    # exe_rnj(topo_num=topo_num)
    # exe_sim(topo_num=topo_num)
    # topo_infer(topo_num=topo_num,bw=1000,probe_num=probe_num)
    # compute_similarity(topo_num=topo_num)
if __name__ == "__main__":
    # main()
    multi_probe()