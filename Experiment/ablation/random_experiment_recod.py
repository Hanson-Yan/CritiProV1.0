# random_experiment_record.py - 精简版（仅保留 unrealized links ratio）

import sys
sys.path.append(f"/home/retr0/Project/TopologyObfu/CritiPro/topo_deployment")

import numpy as np
import os
import json
from adjacency_to_routing import run_ad_to_rout, get_user_input


class ProTODataLoader:
    """
    专门用于读取 ProTO 随机混淆拓扑的数据加载器
    """
    def __init__(self, proto_base_dir="/home/retr0/Project/TopologyObfu/ProTO"):
        self.proto_base_dir = proto_base_dir
    
    def load_data(self, topo_num):
        """
        从 ProTO 目录加载数据
        :return: confuse_adj_path
        """
        topo_result_dir = os.path.join(self.proto_base_dir, f"{topo_num}_result", "output_file")
        
        # 检查目录是否存在
        if not os.path.exists(topo_result_dir):
            raise FileNotFoundError(f"ProTO 结果目录不存在: {topo_result_dir}")
        
        # 文件路径
        confuse_adj_path = os.path.join(topo_result_dir, "topo_matrix_confuse.txt")
        
        # 验证文件存在性
        if not os.path.exists(confuse_adj_path):
            raise FileNotFoundError(f"ProTO 文件不存在: {confuse_adj_path}")
        
        print(f"  [ProTO] 成功加载数据:")
        print(f"    Confuse Adj: {confuse_adj_path}")
        
        return confuse_adj_path


class AblationExperimentRecorder:
    def __init__(self, output_dir="./ablation_results"):
        """
        消融实验记录器
        :param output_dir: 结果保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据路径
        self.data_deplo_dir = "/home/retr0/Project/TopologyObfu/CritiPro/topo_obfuscation_ccs/data_back/data_deplo"
        
        # 原始拓扑数据根目录
        self.original_topo_dir = "/home/retr0/Project/TopologyObfu/CritiPro"
        
        # ProTO 数据加载器
        self.proto_loader = ProTODataLoader()
    
    def calculate_unrealizable_ratio(self, original_adj_path, confuse_adj_path):
        """
        计算物理不可实现链路比例
        
        定义：混淆拓扑中,有多少边的跳数 < 原始拓扑中的最短路径跳数
        
        :return: (unrealizable_count, total_edges, ratio)
        """
        try:
            # 读取邻接矩阵
            G_original = np.loadtxt(original_adj_path)
            G_confuse = np.loadtxt(confuse_adj_path)
            
            n = G_original.shape[0]
            
            # 计算原始拓扑的最短路径（跳数）
            def floyd_warshall_hops(adj_matrix):
                """Floyd-Warshall 算法计算最短跳数"""
                n = adj_matrix.shape[0]
                dist = np.full((n, n), np.inf)
                
                # 初始化：直接相连的边跳数为1
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            dist[i][j] = 0
                        elif adj_matrix[i][j] > 0:  # 有边
                            dist[i][j] = 1
                
                # Floyd-Warshall
                for k in range(n):
                    for i in range(n):
                        for j in range(n):
                            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                
                return dist
            
            hop_original = floyd_warshall_hops(G_original)
            hop_confuse = floyd_warshall_hops(G_confuse)
            
            # 统计不可实现的边
            unrealizable_count = 0
            total_edges = 0
            
            for i in range(n):
                for j in range(i+1, n):  # 只看上三角（无向图）
                    if G_confuse[i][j] > 0:  # 混淆拓扑中存在这条边
                        total_edges += 1
                        
                        # 如果混淆拓扑的跳数 < 原始拓扑的最短跳数
                        if hop_confuse[i][j] < hop_original[i][j]:
                            unrealizable_count += 1
            
            ratio = (unrealizable_count / total_edges * 100) if total_edges > 0 else 0.0
            
            print(f"  [Unrealizable] {unrealizable_count}/{total_edges} edges ({ratio:.2f}%)")
            
            return unrealizable_count, total_edges, ratio
            
        except Exception as e:
            print(f"  [WARNING] 无法计算 Unrealizable Ratio: {e}")
            return 0, 0, 0.0
    
    def run_single_experiment(self, topo_num, prob_num, constraint_type, node_num):
        """
        运行单次实验
        
        constraint_type: "random" (ProTO) 或 "ours" (带约束)
        """
        print(f"\n{'='*60}")
        print(f"Running: {constraint_type.upper()}")
        print(f"Topo: {topo_num}, Prob: {prob_num}, Nodes: {node_num}")
        print(f"{'='*60}")
        
        result = {
            "topo_num": topo_num,
            "prob_num": prob_num,
            "node_num": node_num,
            "constraint_type": constraint_type,
            "unrealizable_count": None,
            "total_edges": None,
            "unrealizable_ratio": None,
            "status": "unknown",
            "error_message": ""
        }
        
        try:
            # 根据类型选择数据源
            if constraint_type == "random":
                # 从 ProTO 加载
                confuse_adj_path = self.proto_loader.load_data(topo_num)
            else:  # "ours"
                # 从 data_deplo 加载
                confuse_adj_path = f"{self.data_deplo_dir}/{topo_num}_output_file/output_adj.txt"
                if not os.path.exists(confuse_adj_path):
                    raise FileNotFoundError(f"文件不存在: {confuse_adj_path}")
            
            # 计算 Unrealizable Ratio
            original_adj_path = f"{self.original_topo_dir}/{topo_num}_result/{topo_num}.txt"
            unreal_count, total_edges, unreal_ratio = self.calculate_unrealizable_ratio(
                original_adj_path, confuse_adj_path
            )
            
            result.update({
                "unrealizable_count": int(unreal_count),
                "total_edges": int(total_edges),
                "unrealizable_ratio": float(unreal_ratio),
                "status": "success"
            })
            
            print(f"\n[Results]")
            print(f"  Unrealizable Ratio: {unreal_ratio:.2f}%")
            
        except Exception as e:
            import traceback
            print(f"[ERROR] {e}")
            print(traceback.format_exc())
            result.update({
                "status": "failed",
                "error_message": str(e)
            })
        
        return result
    
    def get_node_num(self, topo_num):
        """读取节点数量"""
        info_path = f"{self.original_topo_dir}/{topo_num}_result/{topo_num}_info.txt"
        try:
            with open(info_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('switch_num:'):
                        return int(line.split(':')[1].strip())
            return 15
        except:
            return 15
    
    def run_ablation_study(self, topo_list, prob_list):
        """
        运行完整对比实验
        """
        all_results = []
        
        for topo_num in topo_list:
            node_num = self.get_node_num(topo_num)
            print(f"\n{'#'*60}")
            print(f"Processing: {topo_num} (Nodes: {node_num})")
            print(f"{'#'*60}")
            
            for prob_num in prob_list:
                # Random (ProTO)
                result_random = self.run_single_experiment(
                    topo_num, prob_num, "random", node_num
                )
                all_results.append(result_random)
                
                # Ours (带约束)
                result_ours = self.run_single_experiment(
                    topo_num, prob_num, "ours", node_num
                )
                all_results.append(result_ours)
        
        self.save_results(all_results)
        return all_results
    
    def save_results(self, results):
        """保存结果"""
        json_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n[SAVED] {json_path}")
        
        csv_path = os.path.join(self.output_dir, "comparison_results.csv")
        import csv
        if results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        print(f"[SAVED] {csv_path}")


def main():
    recorder = AblationExperimentRecorder(output_dir="./comparison_results")
    
    # 实验配置
    topo_list = ["topo_1", "topo_2", "topo_3", "topo_4"]
    prob_list = [500, 1000, 2000, 3000, 5000, 7000, 10000]
    
    results = recorder.run_ablation_study(topo_list, prob_list)
    
    # 统计
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*60}")
    print(f"Completed: {success}/{len(results)} success")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
