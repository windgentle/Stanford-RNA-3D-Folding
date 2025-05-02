import pandas as pd

# 读取数据
file_path = 'stanford-rna-3d-folding/train_labels_cleaned.csv'
data = pd.read_csv(file_path)

# 假设坐标列为 'x' 和 'y'，计算均值和方差
x_1_mean = data['x_1'].mean()
x_1_std = data['x_1'].std()

y_1_mean = data['y_1'].mean()
y_1_std = data['y_1'].std()

z_1_mean = data['z_1'].mean()
z_1_std = data['z_1'].std()

# 输出结果
print(f"x 坐标均值: {x_1_mean}, 方差: {x_1_std}")
print(f"y 坐标均值: {y_1_mean}, 方差: {y_1_std}")
print(f"z 坐标均值: {z_1_mean}, 方差: {z_1_std}")
