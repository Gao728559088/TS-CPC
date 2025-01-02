import os

from matplotlib import pyplot as plt

# 统计每条轨迹的轨迹点数
def count_trajectory_per_points(data_dir):
    user_dirs=os.listdir(data_dir) # 获取所有用户文件夹
    trajectories_points = {}

    for user_dir in user_dirs:
        user_path=os.path.join(data_dir,user_dir,'Trajectory')
        if os.path.isdir(user_path):
            for plt_file in os.listdir(user_path):
                if plt_file.endswith('.plt'):
                    plt_path=os.path.join(user_path,plt_file)
                    with open(plt_path,'r') as file:
                        lines=file.readlines()
                        num_points=len(lines)-6
                        trajectory_name=f"{user_dir}_{plt_file}"
                        trajectories_points[trajectory_name]=num_points

    return trajectories_points

                    
data_dir="/home/ubuntu/forDataSet/geoLife/Data"
trajectories_points=count_trajectory_per_points(data_dir)

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
count3000 = 0
count4000 = 0
count5000 = 0
count6000 = 0
count7000 = 0
count8000 = 0
count8000_plus = 0
max_points=0
min_points=0
for trajectory,trajectory_count in trajectories_points.items():
            if max_points is None or trajectory_count > max_points:
                max_points = trajectory_count
            if min_points is None or trajectory_count < min_points:
                 min_points = trajectory_count
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
            elif 2000 <= trajectory_count < 3000:
                count3000 += 1
            elif 3000 <= trajectory_count < 4000:
                count4000 += 1
            elif 4000 <= trajectory_count < 5000:
                count5000 += 1
            elif 5000 <= trajectory_count < 6000:
                count6000 += 1
            elif 6000 <= trajectory_count < 7000:
                count7000 += 1
            elif 7000 <= trajectory_count < 8000:
                count8000 += 1
            else:
                count8000_plus += 1

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
print(f"大于2000且小于3000的轨迹数量: {count3000}")
print(f"大于3000且小于4000的轨迹数量: {count4000}")
print(f"大于4000且小于5000的轨迹数量: {count5000}")
print(f"大于5000且小于6000的轨迹数量: {count6000}")
print(f"大于6000且小于7000的轨迹数量: {count7000}")
print(f"大于7000且小于8000的轨迹数量: {count8000}")
print(f"大于8000的轨迹数量: {count8000_plus}")
print(f"最小轨迹点数: {min_points}")
print(f"最大轨迹点数: {max_points}")

