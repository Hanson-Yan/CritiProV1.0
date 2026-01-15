#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOPSIS Top-K vs Actual Top-K Comparison Visualization - English Version

Generates:
1. Per-topology summary (1 figure per topology)
2. Multi-topology overview (1 figure)
3. Text report

Author: Retr0
Date: 2025-01-15
Version: 6.0 (English Only)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

# ==================== Configuration ====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

CONFIG = {
    'output_dir': '/home/retr0/Project/TopologyObfu/Experiment/brute_force_results/',
    'topo_nums': [1, 2, 3, 4],
    'figures_dir': None,
}

CONFIG['figures_dir'] = os.path.join(CONFIG['output_dir'], 'figures/')
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

# ==================== Data Loading ====================

def load_integrated_results(topo_num: int) -> List[Dict]:
    """Load integrated results from CSV"""
    csv_file = os.path.join(CONFIG['output_dir'], 
                           f'integrated_results_{topo_num}.csv')
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        return None
    
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'switch_id': row['switch_id'],
                'topsis_score': float(row['topsis_score']),
                'topsis_rank': int(row['topsis_rank']),
                'impact_score': float(row['impact_score']),
            })
    
    return results

# ==================== Top-K Identification ====================

def identify_topk_sets(results: List[Dict]) -> Tuple[int, Set[str], Set[str], Dict]:
    """
    Identify TOPSIS Top-K and Actual Top-K sets
    
    Returns: (K, topsis_set, actual_set, metrics)
    """
    # TOPSIS Top-K: nodes with topsis_score > 0
    topsis_nodes = [r for r in results if r['topsis_score'] > 0]
    K = len(topsis_nodes)
    topsis_set = {r['switch_id'] for r in topsis_nodes}
    
    # Actual Top-K: top K nodes by impact_score
    sorted_results = sorted(results, key=lambda x: x['impact_score'], reverse=True)
    actual_topk = sorted_results[:K]
    actual_set = {r['switch_id'] for r in actual_topk}
    
    # Compute metrics
    intersection = topsis_set & actual_set
    hit_count = len(intersection)
    precision = hit_count / len(topsis_set) if len(topsis_set) > 0 else 0
    recall = hit_count / len(actual_set) if len(actual_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'intersection': intersection,
        'topsis_only': topsis_set - actual_set,
        'actual_only': actual_set - topsis_set,
        'hit_count': hit_count,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    return K, topsis_set, actual_set, metrics

# ==================== Single Topology Visualization ====================

def plot_single_topo_summary(topo_num: int, K: int, topsis_set: Set[str], 
                             actual_set: Set[str], metrics: Dict):
    """
    Single topology comprehensive figure (3 subplots)
    """
    fig = plt.figure(figsize=(18, 5))
    
    # ===== Subplot 1: Set Comparison =====
    ax1 = plt.subplot(1, 3, 1)
    ax1.axis('off')
    
    intersection_nodes = sorted(metrics['intersection'])
    topsis_only_nodes = sorted(metrics['topsis_only'])
    actual_only_nodes = sorted(metrics['actual_only'])
    
    text_content = f"""
Topology {topo_num} - Node Set Comparison (K={K})

{'='*50}

INTERSECTION ({len(intersection_nodes)} nodes)
{', '.join(intersection_nodes) if intersection_nodes else 'None'}

TOPSIS ONLY ({len(topsis_only_nodes)} nodes)
{', '.join(topsis_only_nodes) if topsis_only_nodes else 'None'}

ACTUAL ONLY ({len(actual_only_nodes)} nodes)
{', '.join(actual_only_nodes) if actual_only_nodes else 'None'}

{'='*50}

Hit Rate: {metrics['hit_count']}/{K} = {metrics['hit_count']/K:.1%}
Precision: {metrics['precision']:.1%}
Recall: {metrics['recall']:.1%}
F1 Score: {metrics['f1_score']:.3f}
    """
    
    ax1.text(0.5, 0.5, text_content, 
            ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Subplot 2: Metrics Bar Chart =====
    ax2 = plt.subplot(1, 3, 2)
    
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax2.bar(range(3), metrics_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(metrics_names, fontsize=11)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title(f'Topology {topo_num} - Evaluation Metrics', 
                 fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='Excellent Threshold')
    ax2.axhline(y=0.6, color='orange', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='Good Threshold')
    ax2.legend(loc='upper right', fontsize=9)
    
    # ===== Subplot 3: Node Distribution Pie Chart =====
    ax3 = plt.subplot(1, 3, 3)
    
    sizes = [len(intersection_nodes), len(topsis_only_nodes), len(actual_only_nodes)]
    labels = [
        f'Intersection ({len(intersection_nodes)})',
        f'TOPSIS Only ({len(topsis_only_nodes)})',
        f'Actual Only ({len(actual_only_nodes)})'
    ]
    colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
    explode = (0.1, 0, 0)
    
    # Filter out zero values
    sizes_filtered = [s for s in sizes if s > 0]
    labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
    colors_filtered = [c for c, s in zip(colors_pie, sizes) if s > 0]
    explode_filtered = [e for e, s in zip(explode, sizes) if s > 0]
    
    if sizes_filtered:
        wedges, texts, autotexts = ax3.pie(
            sizes_filtered, 
            labels=labels_filtered, 
            colors=colors_filtered,
            explode=explode_filtered,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
    
    ax3.set_title(f'Topology {topo_num} - Node Distribution', 
                 fontsize=13, fontweight='bold')
    
    # ===== Save Figure =====
    plt.tight_layout()
    save_path = os.path.join(CONFIG['figures_dir'], f'topology_{topo_num}_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[SAVED] {save_path}")

# ==================== Multi-Topology Overview ====================

def plot_all_topos_summary(all_results: Dict):
    """
    Multi-topology overview (2x2 subplots)
    """
    topo_nums = sorted(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # ===== Subplot 1: F1 Score Comparison =====
    ax1 = axes[0]
    f1_scores = [all_results[t][3]['f1_score'] for t in topo_nums]
    colors = ['#2ecc71' if f1 >= 0.8 else '#f39c12' if f1 >= 0.6 else '#e74c3c' 
              for f1 in f1_scores]
    
    bars = ax1.bar(range(len(topo_nums)), f1_scores, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.axhline(y=0.8, color='green', linestyle='--', linewidth=2, 
               alpha=0.6, label='Excellent (>= 0.8)')
    ax1.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, 
               alpha=0.6, label='Good (>= 0.6)')
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(range(len(topo_nums)))
    ax1.set_xticklabels([f'Topo {t}' for t in topo_nums], fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax1.set_title('F1 Score Comparison Across Topologies', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 2: Precision vs Recall =====
    ax2 = axes[1]
    precisions = [all_results[t][3]['precision'] for t in topo_nums]
    recalls = [all_results[t][3]['recall'] for t in topo_nums]
    
    x = np.arange(len(topo_nums))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, precisions, width, label='Precision', 
                   color='#9b59b6', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, recalls, width, label='Recall', 
                   color='#e67e22', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Topo {t}' for t in topo_nums], fontsize=12)
    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 3: Hit Rate =====
    ax3 = axes[2]
    hit_rates = [all_results[t][3]['hit_count'] / all_results[t][0] for t in topo_nums]
    hit_counts = [all_results[t][3]['hit_count'] for t in topo_nums]
    K_values = [all_results[t][0] for t in topo_nums]
    
    bars = ax3.bar(range(len(topo_nums)), hit_rates, 
                   color='#16a085', alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, (bar, hit, k) in enumerate(zip(bars, hit_counts, K_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{hit}/{k}\n({height:.1%})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylim(0, 1.15)
    ax3.set_xticks(range(len(topo_nums)))
    ax3.set_xticklabels([f'Topo {t}' for t in topo_nums], fontsize=12)
    ax3.set_ylabel('Hit Rate', fontsize=13, fontweight='bold')
    ax3.set_title('TOPSIS Top-K Hit Rate', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # ===== Subplot 4: Summary Table =====
    ax4 = axes[3]
    ax4.axis('off')
    # Build table data
    table_data = []
    for t in topo_nums:
        K, _, _, m = all_results[t]
        table_data.append([
            f'Topo {t}',
            f'{K}',
            f'{m["hit_count"]}/{K}',
            f'{m["precision"]:.2%}',
            f'{m["recall"]:.2%}',
            f'{m["f1_score"]:.3f}'
        ])
    # Add average row
    avg_f1 = np.mean([all_results[t][3]['f1_score'] for t in topo_nums])
    avg_prec = np.mean([all_results[t][3]['precision'] for t in topo_nums])
    avg_rec = np.mean([all_results[t][3]['recall'] for t in topo_nums])
    table_data.append([
        'Average',
        '-',
        '-',
        f'{avg_prec:.2%}',
        f'{avg_rec:.2%}',
        f'{avg_f1:.3f}'
    ])
    col_labels = ['Topology', 'K', 'Hits', 'Precision', 'Recall', 'F1 Score']
    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.16, 0.12, 0.16, 0.18, 0.18, 0.18]  # 调整列宽比例
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)  # 从11增加到13
    table.scale(1.2, 3.5)   # 从(1, 2.5)增加到(1.2, 3.5)，横向和纵向都放大
    # Header style
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)  # 表头字体加大
    # Highlight average row
    for i in range(len(col_labels)):
        table[(len(table_data), i)].set_facecolor('#f39c12')
        table[(len(table_data), i)].set_text_props(weight='bold', fontsize=13)  # 平均行字体也加大
    # 调整表格位置，使其更好地填充右下角空间
    ax4.set_title('Summary Statistics', fontsize=15, fontweight='bold', pad=10)  # 减小pad，标题更靠近表格
    
    # ===== Save Figure =====
    plt.tight_layout()
    save_path = os.path.join(CONFIG['figures_dir'], 'multi_topology_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[SAVED] {save_path}")

# ==================== Report Generation ====================

def generate_report(all_results: Dict):
    """Generate comparison report"""
    report_file = os.path.join(CONFIG['output_dir'], 'topk_comparison_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TOPSIS Top-K vs Actual Top-K Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        avg_f1 = np.mean([m['f1_score'] for _, _, _, m in all_results.values()])
        avg_prec = np.mean([m['precision'] for _, _, _, m in all_results.values()])
        avg_rec = np.mean([m['recall'] for _, _, _, m in all_results.values()])
        
        f.write("OVERALL SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of Topologies: {len(all_results)}\n")
        f.write(f"Average Precision: {avg_prec:.2%}\n")
        f.write(f"Average Recall: {avg_rec:.2%}\n")
        f.write(f"Average F1 Score: {avg_f1:.3f}\n\n")
        
        if avg_f1 >= 0.8:
            rating = "EXCELLENT"
        elif avg_f1 >= 0.6:
            rating = "GOOD"
        else:
            rating = "FAIR"
        
        f.write(f"Overall Rating: {rating}\n\n")
        
        # Per-topology details
        f.write("="*80 + "\n")
        f.write("PER-TOPOLOGY DETAILS\n")
        f.write("="*80 + "\n\n")
        
        for topo_num in sorted(all_results.keys()):
            K, topsis_set, actual_set, m = all_results[topo_num]
            
            f.write(f"Topology {topo_num}:\n")
            f.write(f"  K: {K}\n")
            f.write(f"  TOPSIS Top-K: {sorted(topsis_set)}\n")
            f.write(f"  Actual Top-K: {sorted(actual_set)}\n")
            f.write(f"  Intersection: {sorted(m['intersection'])} ({len(m['intersection'])} nodes)\n")
            f.write(f"  TOPSIS Only: {sorted(m['topsis_only'])} ({len(m['topsis_only'])} nodes)\n")
            f.write(f"  Actual Only: {sorted(m['actual_only'])} ({len(m['actual_only'])} nodes)\n")
            f.write(f"  Hit Rate: {m['hit_count']}/{K} = {m['hit_count']/K:.1%}\n")
            f.write(f"  Precision: {m['precision']:.2%}\n")
            f.write(f"  Recall: {m['recall']:.2%}\n")
            f.write(f"  F1 Score: {m['f1_score']:.3f}\n")
            
            if m['f1_score'] >= 0.8:
                f.write(f"  Rating: EXCELLENT\n")
            elif m['f1_score'] >= 0.6:
                f.write(f"  Rating: GOOD\n")
            else:
                f.write(f"  Rating: FAIR\n")
            
            f.write("\n")
        
        # Conclusion
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if avg_f1 >= 0.8:
            f.write("The TOPSIS method demonstrates EXCELLENT performance in identifying\n")
            f.write("critical nodes, with high agreement between TOPSIS Top-K and actual\n")
            f.write("high-impact nodes.\n")
        elif avg_f1 >= 0.6:
            f.write("The TOPSIS method shows GOOD performance in identifying critical nodes,\n")
            f.write("with reasonable agreement between TOPSIS Top-K and actual high-impact nodes.\n")
        else:
            f.write("The TOPSIS method shows FAIR performance. Consider adjusting the CDF\n")
            f.write("threshold or indicator weights to improve accuracy.\n")
    
    print(f"[SAVED] {report_file}")

# ==================== Main Function ====================

def main():
    """Main function"""
    print("\n" + "="*80)
    print("TOPSIS Top-K vs Actual Top-K Comparison Visualization")
    print("="*80 + "\n")
    
    all_results = {}
    
    for topo_num in CONFIG['topo_nums']:
        print(f"Processing Topology {topo_num}...")
        
        results = load_integrated_results(topo_num)
        if not results:
            continue
        
        K, topsis_set, actual_set, metrics = identify_topk_sets(results)
        if K == 0:
            print(f"  [WARNING] No TOPSIS Top-K nodes found\n")
            continue
        
        print(f"  K = {K}")
        print(f"  Hits: {metrics['hit_count']}/{K} ({metrics['hit_count']/K:.1%})")
        print(f"  F1 Score: {metrics['f1_score']:.3f}\n")
        
        all_results[topo_num] = (K, topsis_set, actual_set, metrics)
        
        # Generate per-topology figure
        plot_single_topo_summary(topo_num, K, topsis_set, actual_set, metrics)
    
    # Generate overview figure
    if len(all_results) > 1:
        print("\nGenerating multi-topology overview...")
        plot_all_topos_summary(all_results)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(all_results)
    
    # Summary
    avg_f1 = np.mean([m['f1_score'] for _, _, _, m in all_results.values()])
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"Topologies Processed: {len(all_results)}")
    print(f"Output Directory: {CONFIG['figures_dir']}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    
    if avg_f1 >= 0.8:
        print(f"Overall Rating: EXCELLENT")
    elif avg_f1 >= 0.6:
        print(f"Overall Rating: GOOD")
    else:
        print(f"Overall Rating: FAIR")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
