
class DelayMatrixProcessor:
    
    def __init__(self, input_file, output_file):
        """
        初始化 DelayMatrixProcessor 类。

        :param input_file: 存储延迟矩阵的输入文件路径
        :param output_file: 用于存储非零延迟元素的输出文件路径
        """

        self.input_file = input_file
        self.output_file = output_file

    def read_delay_matrix(self):
        """
        从输入文件中读取延迟矩阵数据。

        :return: 二维列表形式的延迟矩阵
        """
        matrix = []
        with open(self.input_file, 'r') as file:
            for line in file:
                # 去掉行首行尾的空白字符，然后按空格分割成列表，并将每个元素转换为浮点数
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)
        return matrix

    def extract_non_zero_delays(self):
        """
        提取延迟矩阵中的非零延迟元素，并将它们写入到输出文件中。
        每个非零延迟元素占一行，提取时逐行遍历矩阵。
        """
        # 从输入文件读取延迟矩阵
        delay_matrix = self.read_delay_matrix()

        # 提取非零延迟元素并写入到输出文件
        with open(self.output_file, 'w') as file:
            for row in delay_matrix:  # 逐行遍历矩阵
                for element in row:  # 遍历当前行的每个元素
                    if element != 0.0:  # 判断元素是否为非零（支持浮点数）
                        file.write(f"{element}\n")  # 将非零延迟元素写入文件，每个元素占1行

        print(f"非零延迟元素已提取到文件 {self.output_file} 中")