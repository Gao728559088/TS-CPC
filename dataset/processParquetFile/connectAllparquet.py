import pandas as pd
import glob

# Step 1: 设置包含多个 Parquet 文件的目录路径
parquet_files_path = 'grab/*.parquet'  # 替换为实际路径

# Step 2: 获取所有 Parquet 文件的路径列表
parquet_files = glob.glob(parquet_files_path)

# Step 3: 初始化一个空列表，用于存储每个文件的 DataFrame
df_list = []

# Step 4: 读取每个 Parquet 文件并存储到 df_list 列表中
for file in parquet_files:
    df = pd.read_parquet(file)
    df_list.append(df)

# Step 5: 将所有 DataFrame 合并成一个大的 DataFrame
df_combined = pd.concat(df_list, ignore_index=True)

# Step 6: 将合并后的 DataFrame 保存为一个新的 Parquet 文件
output_path = 'grab1/combined_file.parquet'  # 替换为输出文件的路径
df_combined.to_parquet(output_path, index=False)

print(f"合并后的 Parquet 文件已保存到 {output_path}")
