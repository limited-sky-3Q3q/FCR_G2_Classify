# -*- coding: utf-8 -*-
"""
复现SGD模型90.18%准确率
基于极限优化报告.md中的最佳配置
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("复现SGD模型90.18%准确率")
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
print(f"原始特征数: {len(feature_cols)}")
print(f"类别分布: 类别1={(y==1).sum()}, 类别2={(y==2).sum()}")

# 2. RFE特征选择 - 选择8个特征
print("\n【2. RFE特征选择】")

logistic = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
rfe = RFE(estimator=logistic, n_features_to_select=8, step=1)
X_rfe = rfe.fit_transform(X, y)

selected_features = X.columns[rfe.support_].tolist()

print(f"选择的特征数: {len(selected_features)}")
print(f"选择的特征:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# 3. RobustScaler标准化
print("\n【3. RobustScaler标准化】")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_rfe)

print("标准化完成 (基于中位数和四分位数)")

# 4. 配置SGDClassifier (极限优化报告中的最佳配置)
print("\n【4. 配置SGDClassifier】")
print("使用极限优化报告中的最佳配置:")
print("  loss='log_loss'")
print("  penalty='elasticnet'")
print("  alpha=0.0002")
print("  l1_ratio=0.1")
print("  max_iter=3000")
print("  class_weight='balanced'")

sgd = SGDClassifier(
    loss='log_loss',           # 对数损失 (Logistic损失)
    penalty='elasticnet',      # L1+L2正则化
    alpha=0.0002,             # 学习率倒数 (正则化强度)
    l1_ratio=0.1,             # L1占比10%, L2占比90%
    max_iter=3000,            # 最大迭代次数
    class_weight='balanced',  # 类别平衡
    random_state=42
)

# 5. 8折分层交叉验证
print("\n【5. 8折分层交叉验证】")
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

print(f"交叉验证策略: 8折分层交叉验证")
print(f"样本分布: 类别1={sum(y==1)}, 类别2={sum(y==2)}")

# 6. 执行交叉验证
print("\n【6. 执行交叉验证】")
scores = cross_val_score(sgd, X_scaled, y, cv=skf, scoring='accuracy')

print(f"\n各折准确率:")
for i, score in enumerate(scores, 1):
    print(f"  折{i}: {score*100:.2f}%")

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"\n平均准确率: {mean_accuracy*100:.2f}%")
print(f"标准差: ±{std_accuracy*100:.2f}%")
print(f"准确率范围: [{(mean_accuracy - std_accuracy)*100:.2f}%, {(mean_accuracy + std_accuracy)*100:.2f}%]")

# 7. 结果对比
print("\n" + "="*80)
print("复现结果")
print("="*80)

print(f"\n极限优化报告中的准确率: 90.18%")
print(f"本次复现的准确率: {mean_accuracy*100:.2f}%")

if abs(mean_accuracy - 0.9018) < 0.01:
    print("\n[成功] 成功复现90.18%的准确率！")
    print(f"误差: {abs(mean_accuracy - 0.9018)*100:.2f}%")
else:
    print(f"\n[警告] 复现准确率与报告有差异")
    print(f"差异: {abs(mean_accuracy - 0.9018)*100:.2f}%")
    if mean_accuracy < 0.9018:
        print(f"复现准确率较低: {(0.9018 - mean_accuracy)*100:.2f}%")
    else:
        print(f"复现准确率更高: {(mean_accuracy - 0.9018)*100:.2f}%")

# 8. 输出完整配置
print("\n" + "="*80)
print("完整配置信息")
print("="*80)

print(f"\n数据集:")
print(f"  样本数: {len(data)}")
print(f"  特征数: {len(selected_features)}")
print(f"  类别1: {sum(y==1)}")
print(f"  类别2: {sum(y==2)}")

print(f"\n特征选择:")
print(f"  方法: RFE (递归特征消除)")
print(f"  基估计器: LogisticRegression")
print(f"  选择特征数: 8")
print(f"  特征列表: {', '.join(selected_features)}")

print(f"\n标准化:")
print(f"  方法: RobustScaler")
print(f"  说明: 基于中位数和四分位数,对异常值鲁棒")

print(f"\n模型:")
print(f"  类型: SGDClassifier")
print(f"  loss: log_loss")
print(f"  penalty: elasticnet")
print(f"  alpha: 0.0002")
print(f"  l1_ratio: 0.1")
print(f"  max_iter: 3000")
print(f"  class_weight: balanced")
print(f"  random_state: 42")

print(f"\n交叉验证:")
print(f"  方法: StratifiedKFold")
print(f"  n_splits: 8")
print(f"  shuffle: True")
print(f"  random_state: 42")

print(f"\n性能指标:")
print(f"  平均准确率: {mean_accuracy*100:.2f}%")
print(f"  标准差: ±{std_accuracy*100:.2f}%")
print(f"  各折准确率: {[f'{s*100:.2f}%' for s in scores]}")

print("\n" + "="*80)
print("复现完成")
print("="*80)

# 9. 训练完整模型并保存配置信息（可选）
print("\n【9. 训练完整模型】")
sgd.fit(X_scaled, y)

print(f"\n模型训练完成")
print(f"收敛迭代次数: {sgd.n_iter_}")

# 获取模型系数
coefficients = sgd.coef_[0]
intercept = sgd.intercept_[0]

print(f"\n决策边界:")
print(f"  截距: {intercept:.4f}")
print(f"\n特征系数:")
for i, (feat, coef) in enumerate(zip(selected_features, coefficients), 1):
    print(f"  {i}. {feat}: {coef:.4f}")

# 特征重要性（按系数绝对值排序）
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n特征重要性排序:")
for i, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Abs_Coefficient']:.4f}")

print("\n" + "="*80)
print("复现脚本执行完毕")
print("="*80)
