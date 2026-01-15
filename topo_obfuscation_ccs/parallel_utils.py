# parallel_utils.py

from concurrent.futures import ProcessPoolExecutor


def safe_eval_with_evaluator(args):
    """
    并行评估的安全包装函数
    
    注意：必须在顶层定义才能被 pickle 序列化
    """
    individual, evaluator = args
    try:
        result = evaluator.evaluate(individual)
        if not isinstance(result, tuple) or len(result) != 3:
            print(f"[❌] 非法返回值: {result}")
            return (1.0, 0.0, 100.0)
        return result
    except Exception as e:
        print(f"[❌] 并行评估异常: {e}")
        return (1.0, 0.0, 100.0)


def parallel_evaluate(population, evaluator, max_workers=4):
    """
    并行评估种群适应度
    
    参数:
        population: 待评估的个体列表
        evaluator: 评估器实例
        max_workers: 并行工作进程数
    
    返回:
        适应度列表
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(ind, evaluator) for ind in population]
        results = list(executor.map(safe_eval_with_evaluator, args))
    return results
