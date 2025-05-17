import pandas as pd
import numpy as np
from datetime import datetime

# 1. 读取 MODIS 火点数据
df = pd.read_csv("fire_archive_M-C61_614567.csv")
df["acq_date"] = pd.to_datetime(df["acq_date"])
df = df[df["confidence"] >= 60]  # 过滤低置信度

# 2. 加载坐标点（或生成）
grid = pd.read_csv("grid_coordinates.csv")  # 或使用你生成的点
grid_num = len(grid)

# 3. 设置时间范围
dates = pd.date_range("2024-01-01", "2024-12-31")
labels = []

for d in dates[:-7]:  # 滑窗开始
    sub = df[df["acq_date"].between(d, d + pd.Timedelta(days=6))]  # 7日窗口
    fire_lat = sub["latitude"].values
    fire_lon = sub["longitude"].values
    label_day = []

    for _, row in grid.iterrows():
        lat, lon = row["lat"], row["lon"]
        matches = (np.abs(fire_lat - lat) < 0.05) & (np.abs(fire_lon - lon) < 0.05)
        match = np.mean(matches) if len(matches) > 0 else 0.0  # ✅ 避免 NaN 标签
        label_day.append(match)  # 值范围为 0 ~ 1

    labels.append(label_day)

# 4. 保存标签
labels = np.array(labels).T  # shape: [N点, T序列]
y = labels.mean(axis=1)      # 作为火灾概率
np.save("data/y_final.npy", y)
print("✅ y_final.npy saved. shape:", y.shape)
