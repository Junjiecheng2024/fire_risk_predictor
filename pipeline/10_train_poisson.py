import pandas as pd
import statsmodels.api as sm
import numpy as np
from joblib import dump
import os

# === 读取数据 ===
df = pd.read_csv("../output/grid_time_features_labels.csv")

# === 选择特征列和标签列 ===
feature_cols = ['NDVI', 'NDWI', 'dist_to_city', 'elevation', 'landcover', 'slope']
X = df[feature_cols]
y = df['fire_count']

# === 添加常数项用于截距 ===
X = sm.add_constant(X)

# === 拟合 Poisson 回归模型 ===
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# === 输出模型摘要和评估 ===
print(results.summary())

# === 预测值和评估 ===
df['fire_count_pred'] = results.predict(X)
print("\n平均预测值:", df['fire_count_pred'].mean())
print("总观测火灾数:", y.sum(), "总预测火灾数:", df['fire_count_pred'].sum())

# === 保存结果（可选）===
df.to_csv("../output/grid_time_features_labels_with_pred.csv", index=False)
print("预测结果已保存至 grid_time_features_labels_with_pred.csv")

# === 保存模型 ===
os.makedirs("../model", exist_ok=True)
dump(results, "../model/poisson_model.joblib")
print("✅ 模型已保存至 ../model/poisson_model.joblib")