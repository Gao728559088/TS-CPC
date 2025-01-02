import os

# 指定包含 .plt 文件的目录
directory = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/iosPltFiles'

# 存储每个轨迹的点数
trajectory_counts = {}

# 计数小于100和100到200之间的轨迹数量
count_less_100 = 0
count_100_to_200 = 0
count300 = 0
count400 = 0
count500 = 0
count600 = 0
count700 = 0
count800 = 0
count900 = 0
count1000 = 0

count2000 = 0
count2000_plus = 0
# 遍历目录中的每个文件
for filename in os.listdir(directory):
    if filename.endswith('.plt'):
        file_path = os.path.join(directory, filename)
        
        # 打开文件并计算行数
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 每个文件的有效轨迹点数 = 总行数 - 7
            trajectory_count = len(lines) - 7
            trajectory_counts[filename] = trajectory_count
            
            # 统计点数范围
            if trajectory_count < 100:
                count_less_100 += 1
            elif 100 <= trajectory_count < 200:
                count_100_to_200 += 1
            elif 200 <= trajectory_count < 300:
                count300 += 1
            elif 300 <= trajectory_count < 400:
                count400 += 1
            elif 400 <= trajectory_count < 500:
                count500 += 1
            elif 500 <= trajectory_count < 600:
                count600 += 1
            elif 600 <= trajectory_count < 700:
                count700 += 1
            elif 700 <= trajectory_count < 800:
                count800 += 1
            elif 800 <= trajectory_count < 900:
                count900 += 1
            elif 900 <= trajectory_count < 1000:
                count1000 += 1
            elif 1000 <= trajectory_count < 2000:
                count2000 += 1
            else:
                count2000_plus += 1


# 输出结果
print(f"小于100的轨迹数量: {count_less_100}")
print(f"大于100且小于200的轨迹数量: {count_100_to_200}")
print(f"大于200且小于300的轨迹数量: {count300}")
print(f"大于300且小于400的轨迹数量: {count400}")
print(f"大于400且小于500的轨迹数量: {count500}")
print(f"大于500且小于600的轨迹数量: {count600}")
print(f"大于600且小于700的轨迹数量: {count700}")
print(f"大于700且小于800的轨迹数量: {count800}")
print(f"大于800且小于900的轨迹数量: {count900}")
print(f"大于900且小于1000的轨迹数量: {count1000}")
print(f"大于1000且小于2000的轨迹数量: {count2000}")
print(f"大于2000的轨迹数量: {count2000_plus}")
