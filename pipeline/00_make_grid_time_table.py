import pandas as pd
import numpy as np
import os

# 1. 读取格点基础数据（自动添加 grid_id 并保存新 CSV）
df_grid = pd.read_csv("../data/fire_features_4000m.csv")

if "grid_id" not in df_grid.columns:
    df_grid['grid_id'] = np.arange(len(df_grid))
    df_grid.to_csv("../data/fire_features_4000m_with_id.csv", index=False)
    print("保存加了 grid_id 的 fire_features_4000m_with_id.csv")
else:
    print("fire_features_4000m.csv 已经有 grid_id 字段")

# 2. 设定时间范围
dates = pd.date_range("2024-01-01", "2024-12-31", freq='D')
df_time = pd.DataFrame({'date': dates})

# 3. 笛卡尔积得到所有格点-时间组合
df_grid_time = df_grid[['grid_id']].assign(key=1).merge(df_time.assign(key=1), on='key').drop('key', axis=1)
os.makedirs("../output", exist_ok=True)
df_grid_time.to_csv("../output/grid_time_table.csv", index=False)
print("输出 grid_time_table.csv:", df_grid_time.shape)
