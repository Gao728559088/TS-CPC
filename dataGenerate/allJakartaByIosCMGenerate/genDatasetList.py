import h5py

def extract_datasets_to_list(h5_file, list_file):
    # 打开HDF5文件和输出文本文件
    with h5py.File(h5_file, 'r') as hf, open(list_file, 'w') as file_list:
        # 遍历HDF5文件中的所有键（数据集名）
        def visit_items(name, item):
            if isinstance(item, h5py.Dataset):
                file_list.write(f'{name}\n')
        
        # 递归遍历所有数据集
        hf.visititems(visit_items)

# HDF5文件路径
h5_file_path = 'dataset/grab_possi/grab_possi_Jakarta_all_new/ios/allJakartaByIos.hdf5'

# 生成列表文件
extract_datasets_to_list(h5_file_path, 'dataGenerate/allJakartaByIosCMGenerate/allJakartaCMByIos.txt')
