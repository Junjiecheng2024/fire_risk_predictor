import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import sys
sys.path.append("..")
from models.lstm_model import LSTMModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



def main():
    X = np.load("../data/data/X_final.npy")  # [N, 7, F]
    y = np.load("../data/data/y_final.npy")[:len(X)]  # [N]

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 创建模型
    model = LSTMModel(X.shape[2]).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(100):
        # 训练阶段
        model.train()
        pred_train = model(X_train).squeeze()
        loss = loss_fn(pred_train, y_train)
        loss.backward()
        opt.step()
        opt.zero_grad()

        # 测试阶段评估 AUC 和 Loss
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test).squeeze()
            test_loss = loss_fn(pred_test, y_test).item()
            auc = roc_auc_score(y_test.cpu().numpy(), pred_test.cpu().numpy())
            print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Test Loss={test_loss:.4f}, AUC={auc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "../models/lstm_model_final.pt")
    print("✅ Model training is completed and saved")

if __name__ == "__main__":
    main()