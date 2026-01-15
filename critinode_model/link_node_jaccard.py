import ast
from typing import List, Set, Tuple
import math

def extract_top_nodes(node_file: str) -> List[str]:
    """提取前K个关键节点"""
    nodes = []
    with open(node_file, 'r') as f:
        for line in f:
            if "节点" in line and "得分" in line:
                node = line.split("节点 ")[1].split(":")[0]
                nodes.append(node.strip())
    return nodes

def extract_top_link_nodes(link_file: str, k_links: int) -> Set[str]:
    """提取前k_links条链路涉及的节点"""
    link_nodes = set()
    with open(link_file, 'r') as f:
        count = 0
        for line in f:
            if "链路" in line and "得分" in line and count < k_links:
                link_part = line.split("链路 ")[1].split(": 得分")[0]
                link = ast.literal_eval(link_part)
                link_nodes.update([str(n) for n in link])
                count += 1
    return link_nodes

def jaccard(A: Set[str], B: Set[str]) -> float:
    return len(A & B) / len(A | B) if (A or B) else 0.0

def precision_recall_f1(A: Set[str], B: Set[str]) -> Tuple[float,float,float]:
    prec = len(A & B) / len(A) if A else 0.0
    rec = len(A & B) / len(B) if B else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def overlap_coefficient(A: Set[str], B: Set[str]) -> float:
    """Overlap coefficient = |A ∩ B| / min(|A|, |B|)"""
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))

def main():
    key_links_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/critical_links_only.txt"
    key_nodes_file = "/home/retr0/Project/TopologyObfu/CritiPro/output_file/critical_node_only.txt"

    # 提取集合
    critical_nodes = extract_top_nodes(key_nodes_file)
    k_links = max(1, math.ceil(len(critical_nodes)/2))
    print(f"k_links {k_links}")
    link_nodes = extract_top_link_nodes(key_links_file, k_links)

    set_nodes = set(critical_nodes)

    # 计算指标
    jac = jaccard(set_nodes, link_nodes)
    p, r, f1 = precision_recall_f1(set_nodes, link_nodes)
    overlap = overlap_coefficient(set_nodes, link_nodes)

    # 输出结果
    print(f"关键节点集合: {set_nodes}")
    print(f"关键链路端点集合: {link_nodes}")
    print(f"Jaccard: {jac:.4f}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    print(f"Overlap coefficient: {overlap:.4f}")

    with open("/home/retr0/Project/TopologyObfu/CritiPro/output_file/jaccard_results.txt", "w") as f:
        f.write(f"原始 Jaccard: {jac:.4f}\n")
        f.write(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n")
        f.write(f"Overlap coefficient: {overlap:.4f}\n")

if __name__ == "__main__":
    main()
