import pandas as pd
import numpy as np
import os

# 加载 NDVI CSV
df = pd.read_csv("fire_features_4000m.csv")
df["date"] = pd.to_datetime(df["system:index"].str[:8], format="%Y%m%d")
df = df.sort_values("date")


# 若日期列乱码，转换处理
if not np.issubdtype(df["date"].dtype, np.datetime64):
    df["date"] = pd.to_datetime(df["date"], errors='coerce')

# 丢弃无效数据
df = df.dropna(subset=["date", "NDVI", "NDWI", "dist_to_city", "elevation", "landcover", "slope"])


# 特征标准化（可选）
df[["NDVI", "NDWI"]] = (df[["NDVI", "NDWI"]] - df[["NDVI", "NDWI"]].mean()) / df[["NDVI", "NDWI"]].std()


# 构造滑动窗口序列（7天一个样本）
window = 7
X = []
for i in range(len(df) - window):
    X.append(df[["NDVI", "NDWI", "dist_to_city", "elevation", "landcover", "slope"]].iloc[i:i+window].values)

X = np.array(X)  # shape: [N, 7, 6]
os.makedirs("data", exist_ok=True)
np.save("data/X_train_ndvi.npy", X)

print("✅ X_train_ndvi.npy saved. shape:", X.shape)
