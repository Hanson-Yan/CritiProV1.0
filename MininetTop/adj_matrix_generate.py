def create_adjacency_matrix():
    # 获取文件名
    file_name = input("请输入保存邻接矩阵的文件名（包括.txt扩展名）: ")

    # 获取总的节点个数
    num_nodes = int(input("请输入总的节点个数: "))
    # 初始化邻接矩阵，所有元素初始为0
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # 依次询问每个节点的连接情况
    for i in range(num_nodes):
        print(f"\n节点 {i} 的连接情况：")
        connections = input(f"请输入节点 {i} 连接的节点编号（用空格分隔，如果没有连接请输入q）: ").strip()

        # 如果输入q，表示没有连接
        if connections.lower() == 'q':
            continue

        # 处理输入的连接编号
        connections = connections.split()
        for conn in connections:
            if conn.isdigit():
                j = int(conn)
                if 0 <= j < num_nodes and j != i:  # 防止输入错误的节点编号和自环
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1  # 无向图，需要更新对称位置
                else:
                    print(f"警告：节点 {i} 的输入 '{conn}' 无效或超出范围。")
            else:
                print(f"警告：节点 {i} 的输入 '{conn}' 不是有效数字。")

    # 将邻接矩阵保存到文件
    with open(file_name, "w") as file:
        for row in adjacency_matrix:
            file.write(" ".join(map(str, row)) + "\n")

    print(f"邻接矩阵已成功保存到文件 {file_name} 中。")

def check_symmetry(file_name):
    # 从文件中读取邻接矩阵
    try:
        with open(file_name, "r") as file:
            adjacency_matrix = [list(map(int, line.strip().split())) for line in file]
    except FileNotFoundError:
        print(f"错误：文件 {file_name} 未找到。")
        return
    except ValueError:
        print(f"错误：文件 {file_name} 格式不正确。")
        return

    # 检查矩阵是否对称
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # 只需检查上三角部分
            if adjacency_matrix[i][j] != adjacency_matrix[j][i]:
                print(f"邻接矩阵不对称，具体位置：A[{i}][{j}] = {adjacency_matrix[i][j]}，A[{j}][{i}] = {adjacency_matrix[j][i]}")
                return

    print("邻接矩阵是对称的。")

# 调用函数
create_adjacency_matrix()
# file_name="/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix/topo_4.txt"
# check_symmetry(file_name)