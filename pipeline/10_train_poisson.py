import pandas as pd
import statsmodels.api as sm
import numpy as np
from joblib import dump
import os

# === Load data ===
df = pd.read_csv("../output/grid_time_features_labels.csv")

# === Select feature columns and label column ===
feature_cols = ['NDVI', 'NDWI', 'dist_to_city', 'elevation', 'landcover', 'slope']
X = df[feature_cols]
y = df['fire_count']

# === Add constant term for intercept ===
X = sm.add_constant(X)

# === Fit Poisson regression model ===
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# === Output model summary and evaluation ===
print(results.summary())

# === Predict and evaluate ===
df['fire_count_pred'] = results.predict(X)
print("\nMean predicted value:", df['fire_count_pred'].mean())
print("Total observed fire count:", y.sum(), "Total predicted fire count:", df['fire_count_pred'].sum())

# === Save results (optional) ===
df.to_csv("../output/grid_time_features_labels_with_pred.csv", index=False)
print("Prediction results saved to grid_time_features_labels_with_pred.csv")

# === Save model ===
os.makedirs("../model", exist_ok=True)
dump(results, "../model/poisson_model.joblib")
print("✅ Model saved to ../model/poisson_model.joblib")
