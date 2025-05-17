import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("..")
from models.lstm_model import LSTMModel

# 模型输入
X = np.load("data/X_predict_today.npy")  # shape: [N, 7, 2]
era5_dummy = np.zeros((X.shape[0], X.shape[1], 4))  # shape: [N, 7, 4]
X = np.concatenate([X, era5_dummy], axis=2)         # → shape: [N, 7, 6]

coords = pd.read_csv("data/grid_coordinates.csv")  # [N, lat, lon]

# 加载模型
model = LSTMModel(X.shape[2])
model.load_state_dict(torch.load("../models/lstm_model_final.pt"))
model.eval()

# 预测
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    preds = model(X_tensor).squeeze().numpy()

coords["fire_risk"] = preds
coords.to_csv("data/prediction_today.csv", index=False)
print("✅ 预测完成，结果保存 data/prediction_today.csv")