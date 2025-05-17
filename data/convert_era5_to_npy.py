import xarray as xr
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# 1. 读取 ERA5 文件
instant = xr.open_dataset("era5/data_stream-oper_stepType-instant.nc")
accum = xr.open_dataset("era5/data_stream-oper_stepType-accum.nc")

# 2. 提取变量
temp = instant["t2m"]
dew = instant["d2m"]
u10 = instant["u10"]
v10 = instant["v10"]
sp = instant["sp"]
prec = accum["tp"]

# 3. 获取时间维度
time = instant["valid_time"].values
n_times = len(time)

# 4. 提取平均值（假设是区域均值）
temp_mean = temp.mean(dim=["latitude", "longitude"]).values
dew_mean = dew.mean(dim=["latitude", "longitude"]).values
u10_mean = u10.mean(dim=["latitude", "longitude"]).values
v10_mean = v10.mean(dim=["latitude", "longitude"]).values
sp_mean = sp.mean(dim=["latitude", "longitude"]).values
prec_mean = prec.mean(dim=["latitude", "longitude"]).values

# 5. 合并成特征序列 [时间步, 特征数]
features = np.stack([temp_mean, dew_mean, u10_mean, v10_mean, sp_mean, prec_mean], axis=1)

# 6. 构造 LSTM 输入
time_steps = 7
X = []
for i in range(len(features) - time_steps):
    X.append(features[i:i + time_steps])
X = np.array(X)

np.save("data/X_train_era5.npy", X)
print("✅ 已保存 X_train_era5.npy，shape:", X.shape)
