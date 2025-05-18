import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# 参数

LABEL_CSV = "../data/fire_archive_M-C61_614567.csv"
OUT_CSV = "../output/grid_time_labels.csv"

# 读格点数据
df_grid = pd.read_csv("../data/fire_features_4000m_lonlat.csv")

required_cols = {'grid_id', 'longitude', 'latitude'}
if not required_cols.issubset(df_grid.columns):
    raise ValueError(f"df_grid 必须包含 {required_cols} 列！当前有: {df_grid.columns.tolist()}")

# 读火点标签数据
df_label = pd.read_csv(LABEL_CSV)

# 火点经纬度，变成数组
label_lon = df_label['longitude'].values
label_lat = df_label['latitude'].values

# 建立KDTree加速匹配
tree = cKDTree(df_grid[['longitude', 'latitude']].values)

# 设置匹配半径（度，视具体情况调整，0.05约等于5km）
radius = 0.05

# 找每个火点最近的格点
dist, idx = tree.query(np.stack([label_lon, label_lat], axis=1), distance_upper_bound=radius)

# 标记属于哪个格点
label_grid_ids = df_grid['grid_id'].values[idx]
df_label['grid_id'] = label_grid_ids

# 过滤没有匹配上的（大于半径的都设置为len(df_grid)，超界）
mask = idx < len(df_grid)
df_label = df_label[mask]

# 输出：每个格点每一天有多少火点
df_label['date'] = pd.to_datetime(df_label['acq_date'])  # acq_date为日期字段
df_label['date'] = df_label['date'].dt.strftime('%Y-%m-%d')

group = df_label.groupby(['grid_id', 'date']).size().reset_index(name='fire_count')
group.to_csv(OUT_CSV, index=False)
print(f"已输出聚合火点数: {OUT_CSV}  共{len(group)}条")
