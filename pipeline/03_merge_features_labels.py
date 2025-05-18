import pandas as pd

# 读取特征文件
df_feat = pd.read_csv("../output/grid_time_features.csv")

# 读取标签文件
df_label = pd.read_csv("../output/grid_time_labels.csv")

# 合并数据：基于 grid_id 和 date
df = df_feat.merge(df_label, on=["grid_id", "date"], how="left")

# 替换 NaN 为 0，并转为整数类型
df["fire_count"] = df["fire_count"].fillna(0).astype(int)

# 保存结果
df.to_csv("../output/grid_time_features_labels.csv", index=False)
print("输出 grid_time_features_labels.csv:", df.shape)
