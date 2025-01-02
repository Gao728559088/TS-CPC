import h5py
import numpy as np
import random

# 用于从 HDF5 文件中加载指定ID的轨迹数据。
def load_trajectory(hdf5_file, trajectory_id):
    with h5py.File(hdf5_file, 'r') as f:
        trajectory = f[trajectory_id][:]
    return trajectory

# 用于获取HDF5文件中所有轨迹的ID，singapore.hdf5 中没有组，直接获取根目录下的键。
def get_all_trajectory_ids(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        return list(f.keys())

# 用于生成轨迹子集。
def generate_subsets(hdf5_file, n, num_subsets):
    # 获取所有轨迹的ID
    all_trajectory_ids = get_all_trajectory_ids(hdf5_file)
    print(f"Total number of trajectories: {len(all_trajectory_ids)}")
    subsets = []
    for _ in range(num_subsets):
        # 随机抽取n条轨迹的ID，生成子集A
        subset_A_ids = random.sample(all_trajectory_ids, n)
        subset_A = []
        subset_Aa = []
        subset_Ab = []
        
        for traj_id in subset_A_ids:
            trajectory = load_trajectory(hdf5_file, traj_id)
            
            # 将轨迹添加到子集A
            subset_A.append(trajectory)
            
            # 生成子集Aa和Ab
            trajectory_a = trajectory[::2]  # 奇数轨迹点
            trajectory_b = trajectory[1::2]  # 偶数轨迹点
            
            subset_Aa.append(trajectory_a)
            subset_Ab.append(trajectory_b)
        
        subsets.append((subset_A, subset_Aa, subset_Ab, subset_A_ids))
    
    return subsets

# 用于将子集保存到新的HDF5文件中。
def save_subsets_to_hdf5(subsets, output_file_base):
    for i, (subset_A, subset_Aa, subset_Ab, subset_A_ids) in enumerate(subsets):
        output_file = f"{output_file_base}_{i+1}.hdf5"
        with h5py.File(output_file, 'w') as f:
            # j 是轨迹ID的索引，traj_id 是轨迹的唯一标识符。
            for j, traj_id in enumerate(subset_A_ids):
                # 直接在根目录下保存数据集，避免冲突，使用唯一名称。
                f.create_dataset(f'{traj_id}_A', data=subset_A[j])
                f.create_dataset(f'{traj_id}_Aa', data=subset_Aa[j])
                f.create_dataset(f'{traj_id}_Ab', data=subset_Ab[j])
                
        # 为当前HDF5文件生成对应的列表文件
        generate_list_from_h5(output_file, f"{output_file_base}_{i+1}.txt")

# 与genDatasetList.py中的函数相同
def generate_list_from_h5(h5_file, list_file):
    def recurse_keys(group, prefix=''):
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                path = f'{prefix}/{key}' if prefix else key
                file_list.write(path + '\n')

    with h5py.File(h5_file, 'r') as hf, open(list_file, 'w') as file_list:
        recurse_keys(hf)

def main():
    hdf5_file = 'dataset/grab_possi/grab_possi_Jakarta_ios/jakartaIos.hdf5'  # 替换为HDF5文件的路径
    output_file_base = 'evalSimilary/jakartaIosSimilary/subsetForVal/singapore_subset'  # 输出文件的基本名称
    n = 3000  # 随机抽取的轨迹数量
    num_subsets = 5  # 生成子轨迹的条数

    subsets = generate_subsets(hdf5_file, n, num_subsets)
    
    # 保存子集到新的HDF5文件，并生成对应的列表文件
    save_subsets_to_hdf5(subsets, output_file_base)
    
    # 打印一些信息以验证结果
    for i, (subset_A, subset_Aa, subset_Ab, _) in enumerate(subsets):
        print(f"Subset {i+1} A contains {len(subset_A)} trajectories")
        print(f"Subset {i+1} Aa contains {len(subset_Aa)} trajectories")
        print(f"Subset {i+1} Ab contains {len(subset_Ab)} trajectories")

        for j in range(len(subset_A)):
            print(f"Trajectory {j} in Subset {i+1} A has {subset_A[j].shape[0]} points")
            print(f"Trajectory {j} in Subset {i+1} Aa has {subset_Aa[j].shape[0]} points")
            print(f"Trajectory {j} in Subset {i+1} Ab has {subset_Ab[j].shape[0]} points")

if __name__ == "__main__":
    main()
