# nsga2_solver.py - Min(关键性) + Min(成本) 版本

from deap import base, creator, tools, algorithms
import numpy as np
import random


class NSGA2Solver:
    def __init__(self, evaluator, individual_length, population_size=100, generations=100, 
                 crossover_prob=0.9, mutation_prob=0.2):
        """
        NSGA-II 求解器 - Min(关键性) + Min(成本)
        
        目标权重:
        - 目标1: Min key_score_avg  → weight = -1.0
        - 目标2: Min cost           → weight = -1.0
        - 目标3: Min penalty        → weight = -1.0
        """
        self.evaluator = evaluator
        self.length = individual_length
        self.population_size = population_size
        self.generations = generations
        self.cxpb = crossover_prob
        self.mutpb = mutation_prob
        
        self.toolbox = self._setup_deap()

    def _setup_deap(self):
        """设置 DEAP"""
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # 三个目标都是最小化
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", lambda: random.randint(0, 1))
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def _get_valid_solutions_with_fitness(self, pop):
        """提取有效解"""
        valid_solutions = []
        
        for ind in pop:
            fit = ind.fitness.values
            # 有效解：不是极差解
            if fit not in [(1.0, 1.0, 1000.0), (1.0, 1.0, 100.0)]:
                valid_solutions.append((ind, fit))
        
        return valid_solutions

    def _get_top_valid_solutions(self, pop, top_k=3):
        """
        获取前 top_k 个最优有效解
        
        排序: penalty 升序 → key_score 升序 → cost 升序
        """
        valid_solutions = self._get_valid_solutions_with_fitness(pop)
        
        if not valid_solutions:
            return []
        
        sorted_solutions = sorted(
            valid_solutions,
            key=lambda x: (x[1][2], x[1][0], x[1][1])  # (penalty, key_score, cost)
        )
        
        top_solutions = [(fit, rank+1) for rank, (ind, fit) in enumerate(sorted_solutions[:top_k])]
        
        return top_solutions

    def _print_generation_info(self, gen, pop, fitnesses):
        """打印当前代信息"""
        print(f"\n{'='*70}")
        print(f"[第 {gen+1}/{self.generations} 代]")
        
        # 统计有效解
        valid_count = sum(1 for fit in fitnesses 
                         if fit not in [(1.0, 1.0, 1000.0), (1.0, 1.0, 100.0)])
        
        # 获取前3个最优有效解
        top_valid = self._get_top_valid_solutions(pop, top_k=3)
        
        # 打印示例
        if top_valid:
            print(f"  [前3个最优有效解]")
            for fit, rank in top_valid:
                # 反归一化成本（假设可修改边数已知）
                actual_cost = int(fit[1] * self.evaluator.n_modifiable_edges)
                print(f"    #{rank}: KeyScore={fit[0]:.4f}, Cost={actual_cost}边({fit[1]:.4f}), Penalty={fit[2]:.1f}")
        else:
            print(f"  [前3个最优有效解] 无有效解")
        
        # 打印有效解比例
        print(f"\n  有效解: {valid_count}/{len(pop)} ({valid_count/len(pop)*100:.1f}%)")

    def run(self, parallel_evaluate_fn=None):
        """执行 NSGA-II"""
        pop = self.toolbox.population(n=self.population_size)
        
        for gen in range(self.generations):
            # ========== 评估 ==========
            if parallel_evaluate_fn:
                fitnesses = parallel_evaluate_fn(pop)
            else:
                fitnesses = [self.evaluator.evaluate(ind) for ind in pop]

            for ind, fit in zip(pop, fitnesses):
                if not isinstance(fit, tuple) or len(fit) != 3:
                    ind.fitness.values = (1.0, 1.0, 100.0)
                else:
                    ind.fitness.values = fit
            
            # ========== 打印信息 ==========
            should_print = (gen == 0 or 
                          gen == self.generations - 1 or 
                          (gen + 1) % 10 == 0)
            
            if should_print:
                self._print_generation_info(gen, pop, fitnesses)
            
            # ========== 进化 ==========
            offspring = algorithms.varAnd(pop, self.toolbox, self.cxpb, self.mutpb)

            # 连通性检查
            valid_offspring = []
            for ind in offspring:
                matrix = self.evaluator.encoder.decode(ind)
                if self.evaluator._is_connected(matrix):
                    valid_offspring.append(ind)

            offspring = valid_offspring

            # 评估 offspring
            if parallel_evaluate_fn:
                fitnesses_off = parallel_evaluate_fn(offspring)
            else:
                fitnesses_off = [self.evaluator.evaluate(ind) for ind in offspring]

            for ind, fit in zip(offspring, fitnesses_off):
                if not isinstance(fit, tuple) or len(fit) != 3:
                    ind.fitness.values = (1.0, 1.0, 100.0)
                else:
                    ind.fitness.values = fit

            # ========== 选择 ==========
            pop = self.toolbox.select(pop + offspring, self.population_size)

        # ========== 提取 Pareto 前沿 ==========
        best_solutions = tools.sortNondominated(pop, k=self.population_size, first_front_only=True)[0]
        
        best_solutions = [ind for ind in best_solutions 
                         if ind.fitness.values not in [(1.0, 1.0, 1000.0), (1.0, 1.0, 100.0)]]
        
        # ========== 最终总结 ==========
        print(f"\n{'='*70}")
        print(f"[NSGA-II 完成]")
        print(f"  Pareto 前沿: {len(best_solutions)} 个有效解")
        print(f"  执行代数: {self.generations}")
        
        final_stats = self.evaluator.get_statistics()
        print(f"\n  [累计统计]")
        print(f"    总评估: {final_stats['evaluations']}")
        print(f"    可行解: {final_stats['feasible_count']}")
        print(f"    比例: {final_stats['feasible_ratio']:.2%}")
        
        # 打印前5个最优解
        if best_solutions:
            print(f"\n  [Pareto 前沿前5个最优解]")
            top_pareto = self._get_top_valid_solutions(best_solutions, top_k=5)
            for fit, rank in top_pareto:
                actual_cost = int(fit[1] * self.evaluator.n_modifiable_edges)
                print(f"    #{rank}: KeyScore={fit[0]:.4f}, Cost={actual_cost}边, Penalty={fit[2]:.1f}")
        
        print(f"{'='*70}\n")
        
        return best_solutions
