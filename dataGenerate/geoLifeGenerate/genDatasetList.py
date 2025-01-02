import h5py
"""
/ (Root group)
├── group1
│   ├── dataset1
│   └── dataset2
├── group2
│   ├── group3
│   │   └── dataset3
└── dataset4 
"""
# 一个HDF5文件中提取所有数据集的路径，并将这些路径写入一个文本文件
def generate_list_from_h5(h5_file, list_file):
    # 定义一个内部递归函数，用于递归遍历 HDF5 文件中的所有组和数据集。
    def recurse_keys(group, prefix=''):
        # 遍历当前组中的所有键（组名或数据集名）
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                path = f'{prefix}/{key}' if prefix else key
                file_list.write(path + '\n')
            elif isinstance(group[key], h5py.Group):
                new_prefix = f'{prefix}/{key}' if prefix else key
                recurse_keys(group[key], prefix=new_prefix)

    with h5py.File(h5_file, 'r') as hf, open(list_file, 'w') as file_list:
        recurse_keys(hf)

# HDF5文件路径
train_h5 = 'dataset/geoLife/processedGeolife.hdf5'

# 生成列表文件
generate_list_from_h5(train_h5, 'dataGenerate/geoLifeGenerate/geoLife.txt')

