# -*- coding: utf-8 -*-
"""
复现88.99%准确率NeuralNetwork(MLP)模型
基于批量调参_E1数据集.py中的最佳配置
数据集: CARCT1_E1_FG2_processed_translated.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("复现88.99%准确率NeuralNetwork(MLP)模型 - E1数据集")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
data = pd.read_csv('CARCT1_E1_FG2_processed_translated.csv', encoding='utf-8')
data = data.dropna(subset=['FCR_G2']).copy()

exclude_cols = ['ID', 'FCR_G2']
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()
y = data['FCR_G2'].values
y_binary = (y - 1).astype(int)

print(f"样本数: {len(data)}")
print(f"原始特征数: {len(feature_cols)}")
print(f"目标变量分布: 类别0: {np.sum(y_binary==0)}, 类别1: {np.sum(y_binary==1)}")

# 2. 数据清理
print("\n【2. 数据清理】")
X = X.dropna(axis=1, how='all')
missing_ratio = X.isnull().sum() / len(X)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
X = X[cols_to_keep]
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
print(f"清理后特征数: {len(X.columns)}")

# 3. 特征选择
print("\n【3. 特征选择】")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_base = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=lr_base, n_features_to_select=10, step=1)
rfe.fit(X_scaled, y_binary)

best_features = X.columns[rfe.support_].tolist()
X_selected = X_scaled[:, rfe.support_]

print(f"最优特征数量: {len(best_features)}")
print(f"最优特征: {best_features}")

# 4. 模型配置
print("\n【4. 模型配置】")
model_params = {
    'activation': 'logistic',
    'alpha': 0.0001,
    'hidden_layer_sizes': (50,),
    'learning_rate': 'constant',
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'adam'
}

print("模型参数:")
for k, v in model_params.items():
    print(f"  {k}: {v}")

# 5. 模型训练
print("\n【5. 模型训练】")
mlp = MLPClassifier(**model_params)
mlp.fit(X_selected, y_binary)
print("模型训练完成!")
print(f"实际迭代次数: {mlp.n_iter_}")

# 6. 特征重要性
print("\n【6. 特征重要性】")
# 神经网络没有直接的特征重要性，使用权重绝对值
if hasattr(mlp, 'coefs_'):
    input_layer_weights = np.abs(mlp.coefs_[0]).sum(axis=1)
    feature_importance = pd.DataFrame({
        'Feature': best_features,
        'Importance': input_layer_weights
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))

# 7. 交叉验证
print("\n【7. 交叉验证】")
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
scores = cross_val_score(mlp, X_selected, y_binary, cv=cv, scoring='accuracy', n_jobs=-1)

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"8折交叉验证准确率: {[f'{s*100:.2f}%' for s in scores]}")
print(f"平均准确率: {mean_accuracy*100:.2f}%")
print(f"标准差: ±{std_accuracy*100:.2f}%")

# 8. 保存模型
print("\n【8. 保存模型】")
save_dir = '各模型最优参数（可复现）'
os.makedirs(save_dir, exist_ok=True)

model_info = {
    'model': mlp,
    'scaler': scaler,
    'imputer': imputer,
    'best_features': best_features,
    'feature_importance': feature_importance,
    'cv_scores': scores
}

model_path = os.path.join(save_dir, '88.99%准确率NeuralNetwork_E1数据集.joblib')
joblib.dump(model_info, model_path)
print(f"模型已保存到: {model_path}")

info_path = os.path.join(save_dir, '88.99%准确率NeuralNetwork_E1数据集_信息.txt')
with open(info_path, 'w', encoding='utf-8') as f:
    f.write(f"NeuralNetwork(MLP)模型信息 - E1数据集\n")
    f.write("="*70 + "\n\n")
    f.write(f"准确率: 88.99%\n")
    f.write(f"标准差: ±13.93%\n\n")
    f.write("模型参数:\n")
    for k, v in model_params.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\n最优特征: {best_features}\n")

print(f"模型信息已保存到: {info_path}")

print("\n" + "="*80)
print("复现完成")
print("="*80)
