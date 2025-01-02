import h5py
import numpy as np

# 从 HDF5 文件中加载指定ID的轨迹数据
def load_trajectory(hdf5_file, trajectory_id):
    with h5py.File(hdf5_file, 'r') as f:
        trajectory = f[trajectory_id][:]
    return trajectory

# 从 txt 文件中读取轨迹ID
def load_trajectory_ids_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        trajectory_ids = f.read().splitlines()
    return trajectory_ids

# 生成轨迹子集
def generate_subsets_from_txt(hdf5_file, txt_file):
    # 从 txt 文件中读取轨迹ID
    trajectory_ids = load_trajectory_ids_from_txt(txt_file)
    print(f"Number of trajectories in list: {len(trajectory_ids)}")
    
    subset_A = []
    subset_Aa = []
    subset_Ab = []

    for traj_id in trajectory_ids:
        trajectory = load_trajectory(hdf5_file, traj_id)
        
        # 将轨迹添加到子集A
        subset_A.append(trajectory)
        
        # 生成子集Aa和Ab
        trajectory_a = trajectory[::2]  # 奇数轨迹点
        trajectory_b = trajectory[1::2]  # 偶数轨迹点
        
        subset_Aa.append(trajectory_a)
        subset_Ab.append(trajectory_b)

    return (subset_A, subset_Aa, subset_Ab, trajectory_ids)

# 将子集保存到新的HDF5文件中
def save_subsets_to_hdf5(subset_A, subset_Aa, subset_Ab, trajectory_ids, output_file):
    with h5py.File(output_file, 'w') as f:
        for j, traj_id in enumerate(trajectory_ids):
            f.create_dataset(f'{traj_id}_A', data=subset_A[j])
            f.create_dataset(f'{traj_id}_Aa', data=subset_Aa[j])
            f.create_dataset(f'{traj_id}_Ab', data=subset_Ab[j])

    # 生成对应的列表文件
    generate_list_from_h5(output_file, output_file.replace('.hdf5', '.txt'))

# 生成HDF5文件的键列表
def generate_list_from_h5(h5_file, list_file):
    def recurse_keys(group, prefix=''):
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                path = f'{prefix}/{key}' if prefix else key
                file_list.write(path + '\n')

    with h5py.File(h5_file, 'r') as hf, open(list_file, 'w') as file_list:
        recurse_keys(hf)

def main():
    hdf5_file = 'dataset/grab_possi/grab_possi_Jakarta_android/jakartaAndroid.hdf5'  # HDF5文件路径
    txt_file = 'dataGenerate/jakartaAndroidGenerate/jakartaAndroidTest.txt'  # 包含轨迹ID的txt文件
    output_file = 'evalSimilary/jakartaAndroidByShareEncoderSimilary/subsetForVal/jakarta_subset.hdf5'  # 输出文件路径
    
    # 从txt文件生成轨迹子集
    subset_A, subset_Aa, subset_Ab, trajectory_ids = generate_subsets_from_txt(hdf5_file, txt_file)
    
    # 保存子集到新的HDF5文件中
    save_subsets_to_hdf5(subset_A, subset_Aa, subset_Ab, trajectory_ids, output_file)

    # 打印一些信息以验证结果
    print(f"Subset A contains {len(subset_A)} trajectories")
    print(f"Subset Aa contains {len(subset_Aa)} trajectories")
    print(f"Subset Ab contains {len(subset_Ab)} trajectories")

    for j in range(len(subset_A)):
        print(f"Trajectory {j} in Subset A has {subset_A[j].shape[0]} points")
        print(f"Trajectory {j} in Subset Aa has {subset_Aa[j].shape[0]} points")
        print(f"Trajectory {j} in Subset Ab has {subset_Ab[j].shape[0]} points")

if __name__ == "__main__":
    main()
