import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = np.load("X_final.npy")
y = np.load("y_final.npy")
print('正样本数:', y.sum(), '总数:', len(y), '比例:', np.mean(y))

# 如果样本量太大，抽样10000个
idx = np.arange(len(y))
np.random.shuffle(idx)
idx = idx[:10000]
X_ = X[idx].reshape(len(idx), -1)
y_ = y[idx]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_, y_)
y_pred = clf.predict_proba(X_)[:,1]
print("AUC:", roc_auc_score(y_, y_pred))
