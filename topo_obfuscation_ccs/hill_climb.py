# hill_climb.py

import random


def local_hill_climb(individual, evaluator, max_steps=5):
    """
    局部爬山算法
    
    **新目标**：
    - 目标1: Min key_score_avg
    - 目标2: Max similarity
    - 目标3: Min penalty
    """
    current = individual[:]
    best_score = evaluator.evaluate(current)

    for step in range(max_steps):
        candidate = current[:]
        flip_index = random.randint(0, len(candidate) - 1)
        candidate[flip_index] ^= 1
        
        new_score = evaluator.evaluate(candidate)
        
        # **多目标比较（修正版）**
        # 改进条件：
        # 1. 目标1更小
        # 2. 目标1相同且目标2更大
        # 3. 前两者相同且目标3更小
        improved = False
        if new_score[0] < best_score[0]:
            improved = True
        elif new_score[0] == best_score[0]:
            if new_score[1] > best_score[1]:  # 相似度越大越好
                improved = True
            elif new_score[1] == best_score[1] and new_score[2] < best_score[2]:
                improved = True
        
        if improved:
            current = candidate
            best_score = new_score

    return current
