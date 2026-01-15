# objective.py - Min(关键性) + Min(成本) 版本 - 相似度软约束

import numpy as np
import networkx as nx
from typing import List, Dict
from utils.critical_node import identify_key_nodes_from_dict, calculate_node_scores_from_dict
from utils.similarity import graph_similarity
from utils.metrics import VirtualNodeMetrics
from utils.metricCache import MatrixCache


class ObfuscationObjective:
    def __init__(self, original_matrix: np.ndarray, key_nodes: List[str], 
                 node_metrics: Dict, encoder,  
                 b_hop: int = 2, alpha_min: float = 0.6, alpha_max: float = 0.9,
                 latency_penalty_weight: float = 5.0,
                 similarity_penalty_weight: float = 100.0,  # ← 新增：相似度惩罚权重
                 edge_default_weight: float = 1.0,
                 adaptive_overlap: bool = True,
                 overlap_mode: str = 'hard',
                 overlap_penalty_weight: float = 50.0,
                 debug_mode: bool = False):
        """
        目标函数评估器 - Min(关键性) + Min(成本) - 相似度软约束
        
        目标:
            目标1: Min key_score_avg  (安全性)
            目标2: Min cost           (部署成本)
        
        约束:
            - 相似度 ∈ [alpha_min, alpha_max] → 软约束（惩罚）
            - 关键节点重合 = 0 (硬约束模式) 或软惩罚
            - 连通性 → 硬约束
            - 物理延迟可行性 → 软约束（惩罚）
        """
        self.original_matrix = original_matrix
        self.n = original_matrix.shape[0]
        self.key_nodes = [str(k) for k in key_nodes]
        self.node_metrics = node_metrics
        self.encoder = encoder
        
        self.b_hop = b_hop
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.latency_penalty_weight = latency_penalty_weight
        self.similarity_penalty_weight = similarity_penalty_weight  # ← 新增
        self.edge_default_weight = edge_default_weight
        self.adaptive_overlap = adaptive_overlap
        self.debug_mode = debug_mode
        
        self.original_graph = nx.from_numpy_array(original_matrix)
        self.sim_cache = MatrixCache()
        self.apsp_matrix = self._compute_apsp(self.original_graph)
        
        # 自适应控制
        self.evaluation_count = 0
        self.feasible_count = 0
        self.allow_overlap = False
        
        # 缓存原始拓扑编码
        self.original_vec = tuple(self.encoder.encode(self.original_matrix))
        
        # 重合模式
        self.overlap_mode = overlap_mode
        self.overlap_penalty_weight = overlap_penalty_weight
        
        # ========== 成本归一化参数 ==========
        self.n_modifiable_edges = len(encoder.edge_list)
        
        # ========== 打印初始化信息 ==========
        print(f"\n{'='*70}")
        print(f"目标函数初始化 - Min(关键性) + Min(成本) - 相似度软约束")
        print(f"{'='*70}")
        print(f"[拓扑信息]")
        print(f"  节点数: {self.n}")
        print(f"  边数: {int(np.sum(original_matrix)/2)}")
        print(f"  关键节点: {self.key_nodes} ({len(self.key_nodes)}个)")
        print(f"  可修改边数: {self.n_modifiable_edges}")
        
        print(f"\n[目标设计]")
        print(f"  目标1: 最小化关键节点平均得分 (安全性)")
        print(f"  目标2: 最小化边修改数量 (部署成本)")
        
        print(f"\n[约束条件]")
        print(f"  相似度范围: [{alpha_min}, {alpha_max}] → 软约束（惩罚权重={similarity_penalty_weight}）")
        print(f"  关键节点重合: {overlap_mode}模式")
        if overlap_mode == 'soft':
            print(f"  重合惩罚权重: {overlap_penalty_weight}")
        print(f"  自适应调整: {'开启' if adaptive_overlap else '关闭'}")
        print(f"  物理延迟检查: 开启 (权重={latency_penalty_weight})")
        
        print(f"\n[调试模式]")
        print(f"  状态: {'开启' if debug_mode else '关闭'}")
        print(f"{'='*70}\n")

    def _compute_apsp(self, graph: nx.Graph) -> np.ndarray:
        """计算全节点对最短路径"""
        n = graph.number_of_nodes()
        apsp = np.full((n, n), np.inf)
        
        try:
            for i in range(n):
                lengths = nx.single_source_shortest_path_length(graph, i)
                for j, dist in lengths.items():
                    apsp[i][j] = dist
        except Exception as e:
            print(f"[警告] APSP 计算异常: {e}")
        
        return apsp

    def _is_connected(self, matrix: np.ndarray) -> bool:
        """检查连通性"""
        try:
            if matrix.shape[0] == 0:
                return False
            return nx.is_connected(nx.from_numpy_array(matrix))
        except:
            return False

    def _check_physical_latency_violation(self, matrix: np.ndarray) -> float:
        """检查物理延迟可行性"""
        violation = 0.0
        edges = np.argwhere(matrix > 0)
        
        for u, v in edges:
            if u >= v:
                continue
            
            logical_latency = self.edge_default_weight
            physical_latency = self.apsp_matrix[u][v]
            
            if np.isinf(physical_latency):
                continue
            
            if logical_latency < physical_latency:
                violation += (physical_latency - logical_latency)
        
        return violation

    def _calculate_cost(self, original_vec: List[int], mod_vec: List[int]) -> int:
        """计算扰动成本（修改的边数）"""
        return sum(o != m for o, m in zip(original_vec, mod_vec))
    
    def _calculate_overlap_penalty(self, overlap: int) -> float:
        """
        计算重合惩罚（指数型）
        
        overlap=0 → penalty=0
        overlap=1 → penalty=50
        overlap=2 → penalty=150
        overlap=3 → penalty=350
        """
        return self.overlap_penalty_weight * (2 ** overlap - 1)

    def _calculate_similarity_penalty(self, similarity: float) -> float:
        """
        计算相似度偏离惩罚（二次型惩罚）
        
        Args:
            similarity: 当前相似度值
        
        Returns:
            penalty: 惩罚值
                - 在区间内: 0
                - 低于下界: weight * (alpha_min - similarity)^2
                - 高于上界: weight * (similarity - alpha_max)^2
        """
        if similarity < self.alpha_min:
            # 低于下界的惩罚（相似度太低）
            deviation = self.alpha_min - similarity
            penalty = self.similarity_penalty_weight * (deviation ** 2)
        elif similarity > self.alpha_max:
            # 高于上界的惩罚（相似度太高，混淆不足）
            deviation = similarity - self.alpha_max
            penalty = self.similarity_penalty_weight * (deviation ** 2)
        else:
            # 在区间内，无惩罚
            penalty = 0.0
        
        return penalty

    def _auto_adjust_overlap(self):
        """自适应调整关键节点重合约束"""
        if not self.adaptive_overlap:
            return
        
        if self.evaluation_count > 0 and self.evaluation_count % 50 == 0:
            feasible_ratio = self.feasible_count / self.evaluation_count
            
            if feasible_ratio < 0.05 and not self.allow_overlap:
                self.allow_overlap = True
                print(f"\n[自适应] 可行解过少 ({feasible_ratio:.2%})，允许最多1个关键节点重合")
            elif feasible_ratio > 0.2 and self.allow_overlap:
                self.allow_overlap = False
                print(f"\n[自适应] 可行解充足 ({feasible_ratio:.2%})，恢复严格模式")

    def evaluate(self, individual: List[int]) -> tuple:
        """
        评估个体适应度
        
        返回: (key_score_avg, normalized_cost, penalty)
            - key_score_avg: 关键节点平均得分 [0, 1]
            - normalized_cost: 归一化成本 [0, 1]
            - penalty: 约束违背惩罚（包括相似度惩罚）
        """
        self.evaluation_count += 1
        
        if self.evaluation_count % 50 == 0:
            self._auto_adjust_overlap()
        
        try:
            matrix = self.encoder.decode(individual)
            penalty = 0.0
            
            # 检查是否是原始拓扑
            current_vec = tuple(individual)
            is_original = (current_vec == self.original_vec)

            # ========== 硬约束: 连通性 ==========
            if not self._is_connected(matrix):
                if self.debug_mode and self.evaluation_count <= 10:
                    print(f"[调试-评估{self.evaluation_count}] 失败: 不连通")
                return (1.0, 1.0, 1000.0)

            # ========== 软约束 1: 相似度边界（改为软约束）==========
            cached_similarity = self.sim_cache.get(self.original_matrix, matrix)
            if cached_similarity is not None:
                similarity = cached_similarity
            else:
                try:
                    similarity = graph_similarity(
                        np.array(self.original_matrix), 
                        np.array(matrix),
                        method='portrait'
                    )
                    
                    if similarity < 0 or similarity > 1 or np.isnan(similarity):
                        if self.debug_mode:
                            print(f"[警告] 相似度异常: {similarity}")
                        similarity = 0.5
                        
                except Exception as e:
                    if self.debug_mode:
                        print(f"[错误] 相似度计算失败: {e}")
                    similarity = 0.0
                
                self.sim_cache.set(self.original_matrix, matrix, similarity)

            # 计算相似度惩罚（取代硬约束）
            similarity_penalty = self._calculate_similarity_penalty(similarity)
            penalty += similarity_penalty
            
            if self.debug_mode and self.evaluation_count <= 20 and similarity_penalty > 0:
                print(f"[调试-评估{self.evaluation_count}] 相似度惩罚: sim={similarity:.4f}, penalty={similarity_penalty:.1f}")

            # ========== 目标2: 修改成本 ==========
            cost = self._calculate_cost(self.encoder.encode(self.original_matrix), individual)
            normalized_cost = cost / self.n_modifiable_edges  # 归一化到 [0, 1]

            # ========== 关键节点识别 ==========
            try:
                metrics_gen = VirtualNodeMetrics(matrix, self.node_metrics)
                virtual_metrics = metrics_gen.get_metrics()
                key_nodes_in_virtual = identify_key_nodes_from_dict(virtual_metrics)
                
                if not isinstance(key_nodes_in_virtual, list):
                    current_keys = []
                else:
                    current_keys = [str(node) for node, _ in key_nodes_in_virtual]
                
                overlap = len(set(current_keys) & set(self.key_nodes))
                
            except Exception as e:
                if self.debug_mode:
                    print(f"[错误] 关键节点识别失败: {e}")
                overlap = 0
            
            # ========== 软约束 2: 关键节点重合 ==========
            if self.overlap_mode == 'hard':
                if not is_original:
                    max_allowed_overlap = 1 if self.allow_overlap else 0
                    if overlap > max_allowed_overlap:
                        if self.debug_mode and self.evaluation_count <= 10:
                            print(f"[调试-评估{self.evaluation_count}] 失败: 重合 ({overlap} > {max_allowed_overlap})")
                        return (1.0, 1.0, 1000.0)
            
            elif self.overlap_mode == 'soft':
                overlap_penalty = self._calculate_overlap_penalty(overlap)
                penalty += overlap_penalty
                
                if self.debug_mode and self.evaluation_count <= 20 and overlap > 0:
                    print(f"[调试-评估{self.evaluation_count}] 重合惩罚: overlap={overlap}, penalty={overlap_penalty:.1f}")

            # ========== 软约束 3: 物理延迟 ==========
            #消融实验对比，注释该约束
            # latency_violation = self._check_physical_latency_violation(matrix)
            # penalty += self.latency_penalty_weight * latency_violation

            # ========== 目标1: 关键节点得分 ==========
            try:
                node_scores = calculate_node_scores_from_dict(virtual_metrics)
                key_score_avg = np.mean([node_scores[k] for k in self.key_nodes if k in node_scores]) if self.key_nodes else 0.0
            except Exception as e:
                if self.debug_mode:
                    print(f"[错误] 关键节点得分失败: {e}")
                key_score_avg = 1.0
            
            # 统计可行解（penalty=0）
            if penalty == 0.0:
                self.feasible_count += 1

            if self.debug_mode and self.evaluation_count <= 10:
                print(f"[调试-评估{self.evaluation_count}] ✓ KeyScore={key_score_avg:.4f}, Cost={cost}({normalized_cost:.4f}), Sim={similarity:.4f}, Penalty={penalty:.1f}")

            return float(key_score_avg), float(normalized_cost), float(penalty)

        except Exception as e:
            if self.debug_mode:
                print(f"[错误-评估{self.evaluation_count}] 异常: {e}")
                import traceback
                traceback.print_exc()
            return (1.0, 1.0, 100.0)

    def get_statistics(self):
        """返回统计信息"""
        feasible_ratio = self.feasible_count / self.evaluation_count if self.evaluation_count > 0 else 0
        return {
            "evaluations": self.evaluation_count,
            "feasible_count": self.feasible_count,
            "feasible_ratio": feasible_ratio,
            "allow_overlap": self.allow_overlap
        }
