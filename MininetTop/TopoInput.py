"""
定义内部交换机网络拓扑数据结构
根据输入的0-1路由矩阵创建网络拓扑的链路信息
"""

# filePath="/home/retr0/Project/TopologyObfu/MininetTop/topo_matrix_1.txt"
# filePath="/home/retr0/Project/TopologyObfu/MininetTop/xiaojia.txt"


#定义拓扑链路信息
class TopoLink:
    def __init__(self, bw=100, delay='20ms', loss=0, max_queue_size=10):
        self.bw = bw
        self.delay = delay
        self.loss = loss
        self.max_queue_size = max_queue_size

    def PrintLinkInfo(self):
        print(f'  Bandwidth is {self.bw},\n'
              f'  Delay is {self.delay},\n'
              f'  Loss is {self.loss},\n'
              f'  Max queue size is {self.max_queue_size}.')
        
#定义拓扑矩阵信息
class TopoMatrix:
    def __init__(self, n):
        """
        初始化矩阵。
        :param n: 矩阵的大小 n×n
        """
        self.n = n
        self.matrix = [[None for _ in range(n)] for _ in range(n)]  # 初始化为 None 的二维列表

    def initialize_matrix(self):
        """
        从用户输入中初始化矩阵。
        """
        print(f"请输入 {self.n}×{self.n} 的 0-1 矩阵，每行用空格分隔：")
        input_matrix = []
        for i in range(self.n):
            while True:
                try:
                    row = list(map(int, input().strip().split()))
                    if len(row) != self.n:
                        raise ValueError("输入的列数不正确，请重新输入。")
                    input_matrix.append(row)
                    break
                except ValueError as e:
                    print(f"输入错误：{e}")
        # 根据用户输入的 0-1 矩阵填充 TopoLink 对象
        for i in range(self.n):
            for j in range(self.n):
                if input_matrix[i][j] == 1:
                    self.matrix[i][j] = TopoLink()  # 在矩阵值为1的位置填充 TopoLink 对象
    
    def initialize_from_file(self, filename):
        """
        从文件中读取 0-1 矩阵并初始化。
        :param filename: 包含矩阵数据的文件名
        """
        try:
            with open(filename, 'r') as file:
                input_matrix = []
                for line in file:
                    row = list(map(int, line.strip().split()))
                    if len(row) != self.n:
                        raise ValueError(f"文件中的行长度不正确，应为 {self.n} 列。")
                    input_matrix.append(row)

                if len(input_matrix) != self.n:
                    raise ValueError(f"文件中的行数不正确，应为 {self.n} 行。")

                for i in range(self.n):
                    for j in range(self.n):
                        if input_matrix[i][j] == 1:
                            self.matrix[i][j] = TopoLink()
        except FileNotFoundError:
            print(f"错误：文件 {filename} 未找到。")
        except ValueError as e:
            print(f"文件格式错误：{e}")
        except Exception as e:
            print(f"读取文件时发生错误：{e}")

    def set_link_property(self, row, col, bw=10, delay='10ms', loss=0, max_queue_size=1000):
        """
        设置矩阵中某个 TopoLink 对象的属性。
        :param row: 行索引
        :param col: 列索引
        :param bw: 带宽
        :param delay: 延迟
        :param loss: 丢包率
        :param max_queue_size: 最大队列大小
        """
        if 0 <= row < self.n and 0 <= col < self.n:
            if self.matrix[row][col] is not None:                
                self.matrix[row][col].bw = bw
                self.matrix[row][col].delay = delay               
                self.matrix[row][col].loss = loss                
                self.matrix[row][col].max_queue_size = max_queue_size
            else:
                print(f"位置 ({row}, {col}) 没有 TopoLink 对象。")
        else:
            raise IndexError("行或列索引超出范围")

    def get_link_property(self, row, col):
        """
        获取矩阵中某个 TopoLink 对象的属性。
        :param row: 行索引
        :param col: 列索引
        :return: 返回一个字典，包含 TopoLink 的所有属性
        """
        if 0 <= row < self.n and 0 <= col < self.n:
            if self.matrix[row][col] is not None:
                return {
                    "bw": self.matrix[row][col].bw,
                    "delay": self.matrix[row][col].delay,
                    "loss": self.matrix[row][col].loss,
                    "max_queue_size": self.matrix[row][col].max_queue_size
                }
            else:
                print(f"位置 ({row}, {col}) 没有 TopoLink 对象。")
                return None
        else:
            raise IndexError("行或列索引超出范围")

    def print_matrix(self):
        """
        打印矩阵的详细信息。
        """
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i][j] is not None:
                    print(f'The link({i},{j}) info:')
                    self.matrix[i][j].PrintLinkInfo()
                else:
                    print(f"位置 ({i}, {j}) 没有 TopoLink 对象。")
            print("-" * 40)  # 分隔不同行的输出

    def enter_modify_mode(self):
        """
        进入修改模式，允许用户修改 TopoLink 的属性。
        """
        print("\n进入修改模式：")
        print("输入 'exit' 退出修改模式。")
        print("输入格式：row col bw delay loss max_queue_size")
        print("示例：0 1 100 5ms 1 500")

        while True:
            command = input("请输入修改指令：").strip()
            if command.lower() == "exit":
                print("退出修改模式。")
                break

            try:
                row, col, bw, delay, loss, max_queue_size = command.split()
                row, col = int(row), int(col)
                bw, loss, max_queue_size = int(bw), int(loss), int(max_queue_size)
                self.set_link_property(row, col, bw=bw, delay=delay, loss=loss, max_queue_size=max_queue_size)
                print(f"位置 ({row}, {col}) 的 TopoLink 属性已更新。")
            except (ValueError, IndexError) as e:
                print(f"输入错误：{e}")

    def get_matrix(self):
        """
        返回当前实例
        """
        return self.matrix

class switchTopoCreator:
    def creatSwitchTopo(num,filePath):
        # 用户输入矩阵大小
        # n = int(input("请输入矩阵的大小 n: "))
        
        # 创建 Matrix 对象
        matrix = TopoMatrix(num)

        # 初始化矩阵
        # matrix.initialize_matrix()
        
        matrix.initialize_from_file(filePath)


        # 打印初始矩阵
        print("初始矩阵:")
        matrix.print_matrix()

        # 调用修改模式
        matrix.enter_modify_mode()

        # 打印修改后的矩阵
        print("\n修改后的矩阵:")
        matrix.print_matrix()

        return matrix

# if __name__== "__main__":
#     topo_matrix = main()
#     print("\n最终矩阵信息：")
#     topo_matrix.print_matrix()