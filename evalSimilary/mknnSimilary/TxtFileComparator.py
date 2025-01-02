class TxtFileComparator:
    def __init__(self, file1, file2):
        """
        初始化文件比较类
        :param file1: 第一个txt文件路径
        :param file2: 第二个txt文件路径
        """
        self.file1 = file1
        self.file2 = file2

    def read_keys(self, file_path):
        """
        从文件中读取所有键（行），并将它们存储到一个集合中
        :param file_path: txt文件路径
        :return: 文件中的所有键（去重，集合类型）
        """
        with open(file_path, 'r') as f:
            keys = f.read().splitlines()  # 按行读取文件内容
        return set(keys)  # 使用集合去重并忽略顺序

    def compare(self):
        """
        比较两个txt文件中的键是否相同
        :return: 如果文件中的键相同，则返回True，否则返回False
        """
        keys1 = self.read_keys(self.file1)
        keys2 = self.read_keys(self.file2)
        
        # 比较两个集合是否相等
        return keys1 == keys2

# 示例用法
file1 = 'evalSimilary/mknnSimilary/transformed_data_distort.txt'  # 第一个txt文件路径
file2 = 'dataGenerate/allSingaporeGenerate/allSingaporeTrain.txt'  # 第二个txt文件路径

comparator = TxtFileComparator(file1, file2)
if comparator.compare():
    print("两个txt文件中的键完全相同")
else:
    print("两个txt文件中的键不同")
