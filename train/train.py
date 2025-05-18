# improved_train.py

import os, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用带权重的BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        prob = torch.sigmoid(inputs)
        pt = prob * targets + (1 - prob) * (1 - targets)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --------------- 路径与模型导入 ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from models.lstm_model import LSTMModel

# --------------- 数据加载与分析 ----------------
X = np.load(os.path.join(ROOT, "data/data/X_final.npy"))
y = np.load(os.path.join(ROOT, "data/data/y_final.npy"))[:len(X)]

# 二值化
y = (y > 0).astype(np.float32)

# 数据分析
print("=== 数据分析 ===")
print(f"总样本数: {len(y)}")
print(f"正样本数: {y.sum():.0f} ({y.mean()*100:.2f}%)")
print(f"负样本数: {len(y) - y.sum():.0f} ({(1-y.mean())*100:.2f}%)")
print(f"数据形状: X={X.shape}, y={y.shape}")

# 如果正样本太少，可能需要调整策略
if y.mean() < 0.01:
    print("⚠️  警告: 正样本比例极低，建议考虑其他方法或调整阈值")

# --------------- 训练参数 ----------------
n_splits = 5
epochs = 100
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= 工具函数 =============
def get_loader(X_arr, y_arr, bs, shuffle=True):
    ds = TensorDataset(torch.tensor(X_arr, dtype=torch.float32),
                       torch.tensor(y_arr, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

def train_epoch(model, loader, loss_fn, opt):
    model.train()
    total, running = 0, 0.0
    all_preds, all_targets = [], []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb).squeeze()
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        
        running += loss.item() * len(xb)
        total += len(xb)
        
        # 收集预测和真实标签用于分析
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return running / total, all_preds, all_targets

def eval_epoch(model, loader, loss_fn):
    model.eval()
    y_true, y_prob = [], []
    running, total = 0.0, 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze()
            prob = torch.sigmoid(logits)
            loss = loss_fn(logits, yb)
            
            running += loss.item() * len(xb)
            total += len(xb)
            y_true.append(yb.cpu().numpy())
            y_prob.append(prob.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    
    # 计算AUC
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    
    return running / total, auc, y_true, y_prob

# ============= K-fold 训练 =============
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_aucs = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🟡 === Fold {fold}/{n_splits} ===")
    
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    # 分析本折数据分布
    print(f"训练集: 正样本 {y_tr.sum():.0f}/{len(y_tr)} ({y_tr.mean()*100:.2f}%)")
    print(f"测试集: 正样本 {y_te.sum():.0f}/{len(y_te)} ({y_te.mean()*100:.2f}%)")
    
    # DataLoader
    train_loader = get_loader(X_tr, y_tr, batch_size, shuffle=True)
    test_loader = get_loader(X_te, y_te, batch_size, shuffle=False)
    
    # 计算pos_weight
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    pos_weight = torch.tensor([neg/pos if pos > 0 else 1.0], device=device)
    print(f"pos_weight: {pos_weight.item():.4f}")
    
    # 模型 & 优化器 & 损失函数
    model = LSTMModel(X.shape[2], hidden_dim=64, num_layers=2, bidir=True).to(device)
    
    # 尝试不同的损失函数
    use_focal = True  # 可以改为False使用BCE
    if use_focal:
        loss_fn = WeightedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 使用更小的学习率和学习率调度器
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10
    )
    
    best_auc = 0.0
    patience_counter = 0
    patience = 20
    current_lr = opt.param_groups[0]['lr']
    
    # -------- 训练循环 ----------
    for epoch in range(1, epochs + 1):
        tr_loss, tr_preds, tr_targets = train_epoch(model, train_loader, loss_fn, opt)
        te_loss, auc, te_targets, te_preds = eval_epoch(model, test_loader, loss_fn)
        
        # 学习率调度
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(te_loss)
        new_lr = opt.param_groups[0]['lr']
        
        # 如果学习率发生变化，打印信息
        if new_lr != old_lr:
            print(f"  学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
        
        # 早停
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(ROOT, f"models/best_lstm_fold{fold}.pt")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        # 打印训练信息
        auc_str = f"{auc:.4f}" if not math.isnan(auc) else "nan"
        print(f"Epoch {epoch:3d}: Train Loss={tr_loss:.4f}, Test Loss={te_loss:.4f}, "
              f"AUC={auc_str}, Best AUC={best_auc:.4f}, LR={new_lr:.6f}")
        
        # 每10个epoch详细分析一次
        if epoch % 10 == 0:
            # 分析预测分布
            print(f"  训练集预测分布: min={tr_preds.min():.4f}, max={tr_preds.max():.4f}, "
                  f"mean={tr_preds.mean():.4f}, std={tr_preds.std():.4f}")
            print(f"  测试集预测分布: min={te_preds.min():.4f}, max={te_preds.max():.4f}, "
                  f"mean={te_preds.mean():.4f}, std={te_preds.std():.4f}")
            
            # 分析预测阈值
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            for thresh in thresholds:
                pred_binary = (te_preds > thresh).astype(int)
                tp = ((pred_binary == 1) & (te_targets == 1)).sum()
                fp = ((pred_binary == 1) & (te_targets == 0)).sum()
                tn = ((pred_binary == 0) & (te_targets == 0)).sum()
                fn = ((pred_binary == 0) & (te_targets == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"    阈值{thresh}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发，在epoch {epoch}")
            break
    
    all_aucs.append(best_auc)
    print(f"✅ Fold {fold} 完成，最佳 AUC: {best_auc:.4f}")

# 总结
print(f"\n🏁 === 训练完成 ===")
print(f"各折AUC: {[f'{auc:.4f}' for auc in all_aucs]}")
print(f"平均AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")

# 如果AUC还是很低，给出建议
if np.mean(all_aucs) < 0.6:
    print("\n⚠️  AUC仍然较低，建议尝试以下方法:")
    print("1. 检查数据质量和特征工程")
    print("2. 尝试不同的模型架构 (CNN, Transformer)")
    print("3. 调整数据预处理方式")
    print("4. 考虑使用更复杂的采样策略")
    print("5. 检查标签是否正确")