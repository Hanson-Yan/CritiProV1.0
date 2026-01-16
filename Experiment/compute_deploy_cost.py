import numpy as np
import os


def cost_count(a, b):
    """
    计算两个向量中对应元素差的绝对值大于1的次数，并累加这些差值的绝对值

    参数:
        a (list): 第一个向量
        b (list): 第二个向量

    返回:
        tuple: (绝对值大于1的次数, 累加的差值绝对值)
    """
    if len(a) != len(b):
        raise ValueError("两个向量的长度必须相同")

    count = 0  # 用于计数
    total_sum = 0  # 用于累加差值的绝对值

    for i in range(len(a)):
        diff = abs(a[i] - b[i])  # 计算对应元素的差的绝对值
        total_sum += diff  # 累加差值的绝对值

        if diff > 0.5:
            count += 1  # 如果绝对值大于1，计数加1
            
    return count,total_sum


def cost_compute(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。

    参数:
        vector_a (numpy.ndarray): 第一个向量
        vector_b (numpy.ndarray): 第二个向量

    返回:
        float: 两个向量的余弦相似度
    """
    # 检查输入是否为 NumPy 数组
    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        raise ValueError("输入必须是 NumPy 数组")

    # 检查向量长度是否一致
    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的长度必须相同")

    # 计算点积
    dot_product = np.dot(vector_a, vector_b)
    
    # 计算模长
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # 防止除以零
    if norm_a == 0 or norm_b == 0:
        raise ValueError("向量的模长不能为零")
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_a * norm_b)
    deploy_cost = (1-cosine_sim)/2
    return deploy_cost  


def compute_deploy_cost(topo_num):
    prob_num_arr=[500,1000,2000,3000,5000,7000,10000]
    model_name=["critipro","proto","antitomo"]
    deploy_cost_dir=f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/deploy_cost"
    for prob_num in prob_num_arr:
        cost_dir=f"{deploy_cost_dir}/{prob_num}.txt"
        # try:
        #     os.makedirs(cost_dir, exist_ok=True)  # 如果文件夹已存在，不会报错
        #     print(f"文件夹 {cost_dir} 已创建成功！")
        # except Exception as e:
        #     print(f"创建文件夹时发生错误：{e}")
        original_vector=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/delay/original/{topo_num}_{prob_num}_delay.txt")
        deploy_cost_arr=[]
        for name in model_name:
            _vector=np.loadtxt(f"/home/retr0/Project/TopologyObfu/Experiment/{topo_num}_result/data/delay/{name}/{topo_num}_{prob_num}_delay.txt")
            # deploy_cost=cost_compute(original_vector,_vector)
            deploy_cost=cost_count(original_vector,_vector)
            deploy_cost_arr.append(deploy_cost)
        deploy_cost_result = {key: value for key, value in zip(model_name, deploy_cost_arr)}
        print(deploy_cost_result)
        with open(cost_dir, "w") as file:
            for key, value in deploy_cost_result.items():
                file.write(f"{key}: {value}\n")


