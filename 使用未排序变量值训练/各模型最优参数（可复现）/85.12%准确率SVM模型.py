# -*- coding: utf-8 -*-
"""
SVM模型 - 85.12%准确率
使用12个特征 + StandardScaler
random_state=42
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SVM模型 - 85.12%准确率")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
data = pd.read_csv('CARCT1_FG2.csv', encoding='gbk')
data = data.dropna(subset=['FCR_G2', 'ID'])

exclude_cols = ['ID', 'FCR_G2', 'time', 'Group',
                'FCR1', 'FCR2', 'FCR3', 'FCR4',
                'PHQ9.1', 'PHQ9.2', 'PHQ9.3', 'PHQ9.4',
                'GAD7.1', 'GAD7.2', 'GAD7.3', 'GAD7.4',
                'ISI1', 'ISI2', 'ISI3', 'ISI4']

feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()
y = data['FCR_G2'].copy().astype(int)

if 'FCR7_0' in X.columns:
    X = X.drop('FCR7_0', axis=1)

print(f"样本数: {len(data)}")
print(f"类别分布: 类别1={(y==1).sum()}, 类别2={(y==2).sum()}")

# 2. 使用12个关键特征
print("\n【2. 使用12个关键特征】")

selected_features = [
    'GAD7_0', 'PHQ9_0', 'ISI_0', 'TCSQ_NC',
    'Emotional_support', 'Stress', 'Exercise',
    'Education', 'Knowingduration_of_cancer',
    'Recognition_disease_severity', 'QLQBR23', 'TCSQ_PC'
]

X_selected = X[selected_features]

print(f"选择的特征数: {len(selected_features)}")

# 3. StandardScaler标准化
print("\n【3. StandardScaler标准化】")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 4. 配置SVM (调优后最优)
print("\n【4. 配置SVM】")
print("最优参数:")
print("  C=0.1")
print("  kernel='sigmoid'")
print("  gamma=0.5")
print("  degree=2")
print("  class_weight='balanced'")
print("  probability=True")
print("  random_state=42")

svm = SVC(
    C=0.1,
    kernel='sigmoid',
    gamma=0.5,
    degree=2,
    class_weight='balanced',
    probability=True,
    random_state=42
)

# 5. 8折分层交叉验证
print("\n【5. 8折分层交叉验证】")
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

scores = cross_val_score(svm, X_scaled, y, cv=skf, scoring='accuracy')

print(f"\n各折准确率:")
for i, score in enumerate(scores, 1):
    print(f"  折{i}: {score*100:.2f}%")

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"\n平均准确率: {mean_accuracy*100:.2f}%")
print(f"标准差: ±{std_accuracy*100:.2f}%")

# 6. 验证结果
print("\n" + "="*80)
print("结果验证")
print("="*80)

target = 0.8512
print(f"\n目标准确率: 85.12%")
print(f"本次准确率: {mean_accuracy*100:.2f}%")
print(f"误差: {abs(mean_accuracy - target)*100:.2f}%")

if abs(mean_accuracy - target) < 0.01:
    print("\n[成功] 复现成功！")
else:
    print("\n[警告] 有差异")

print("\n" + "="*80)
print("配置总结")
print("="*80)

print(f"\n模型: SVM")
print(f"参数: C=0.1, kernel='sigmoid', gamma=0.5, degree=2")
print(f"标准化: StandardScaler")
print(f"特征数: {len(selected_features)}")
print(f"准确率: {mean_accuracy*100:.2f}%")
