def generate_receiver_pairs(n):
    """
    生成接收器对列表。
    :param n: 接收器的总数
    :return: 接收器对列表
    """
    receiver_pairs = []
    
    # 生成 IP 地址格式
    base_ip = "10.0.0."
    
    # 生成所有可能的接收器对
    for i in range(2, n + 2):  # 从接收器 2 开始
        for j in range(i + 1, n + 2):  # 与后续的接收器配对
            receiver_pairs.append((f"{base_ip}{i}", f"{base_ip}{j}"))
    
    return receiver_pairs

print(generate_receiver_pairs(5))
for pair in generate_receiver_pairs(5):
    A,B = pair
    character1 = A.split('.')[-1]
    print(character1)
