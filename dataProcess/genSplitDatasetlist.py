import random
import argparse

"""列表划分"""

def split_dataset(list_file, train_file, valid_file, test_file, train_ratio, valid_ratio, test_ratio):
    with open(list_file, 'r') as file:
        lines = file.readlines()

    # 使用 random.shuffle 方法打乱 lines 列表的顺序，以确保数据随机分布。
    random.shuffle(lines)

    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    valid_end = train_end + int(total_lines * valid_ratio)

    train_data = lines[:train_end]
    valid_data = lines[train_end:valid_end]
    test_data = lines[valid_end:]

    with open(train_file, 'w') as file:
        file.writelines(train_data)

    with open(valid_file, 'w') as file:
        file.writelines(valid_data)

    with open(test_file, 'w') as file:
        file.writelines(test_data)

def main():
    parser = argparse.ArgumentParser(description='Split a dataset list into train, validation and test sets.')
    parser.add_argument('--list_file', type=str, default='dataGenerate/singaporeIosGenerate/singapore.txt',help='DataSetList.txt')
    parser.add_argument('--train_file', type=str, default='dataGenerate/singaporeIosGenerate/singaporeTrain.txt', help='Filename for the training set')
    parser.add_argument('--valid_file', type=str, default='dataGenerate/singaporeIosGenerate/singaporeVal.txt', help='Filename for the validation set')
    parser.add_argument('--test_file', type=str, default='dataGenerate/singaporeIosGenerate/singaporeTest.txt', help='Filename for the test set')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of the dataset to be used for training')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Proportion of the dataset to be used for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proportion of the dataset to be used for testing')

    args = parser.parse_args()

    split_dataset(args.list_file, args.train_file, args.valid_file, args.test_file, args.train_ratio, args.valid_ratio, args.test_ratio)

if __name__ == '__main__':
    main()
