import numpy as np
import os

# 加载 NDVI 和 ERA5 特征序列
X_ndvi = np.load("data/X_train_ndvi.npy")  # shape: [N, 7, 6]
X_era5 = np.load("data/X_train_era5.npy")  # shape: [N, 7, 6]

# 取最短长度对齐
min_len = min(len(X_ndvi), len(X_era5))
X = np.concatenate([X_ndvi[:min_len], X_era5[:min_len]], axis=2)  # 按特征维度拼接

np.save("data/X_final.npy", X)
print("✅ X_final.npy saved. shape:", X.shape)
