# -*- coding: utf-8 -*-
"""
复现92.86%准确率SGD模型
基于调参结果_E1数据集_最终优化.txt中的最佳配置
数据集: CARCT1_E1_FG2_processed_translated.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
import warnings
import joblib
import os

# 获取脚本所在目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(script_dir)

warnings.filterwarnings('ignore')

print("="*80)
print("复现92.86%准确率SGD模型 - E1数据集")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
csv_path = os.path.join(project_dir, 'CARCT1_E1_FG2_processed_translated.csv')
data = pd.read_csv(csv_path, encoding='utf-8')
data = data.dropna(subset=['FCR_G2', 'ID'])

exclude_cols = ['ID', 'FCR_G2']
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()
y = data['FCR_G2'].copy().astype(int)

print(f"样本数: {len(data)}")
print(f"原始特征数: {len(feature_cols)}")

# 2. 数据清理
print("\n【2. 数据清理】")
X = X.dropna(axis=1, how='all')
missing_ratio = X.isnull().sum() / len(X)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
X = X[cols_to_keep]
X = X.fillna(X.median())
print(f"清理后特征数: {len(X.columns)}")

# 3. 特征选择 - 使用RFE选择的10个特征
print("\n【3. 特征选择】")
best_features = [
    'GAD7_0',
    'Residence',
    'Marriage',
    'Education',
    'Partner_Monthly_Income',
    'Relationship_with_Family',
    'Family_Social_Emotional_Support',
    'Chemotherapy',
    'Perceived_Severity_of_Condition',
    'Duration_Aware_of_Cancer_Diagnosis'
]

X_selected = X[best_features]

print(f"选择的特征数: {len(best_features)}")
print(f"选择的特征:")
for i, feat in enumerate(best_features, 1):
    print(f"  {i}. {feat}")

# 4. RobustScaler标准化
print("\n【4. RobustScaler标准化】")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

print("标准化完成 (基于中位数和四分位数)")

# 5. 配置SGDClassifier (最终优化报告中的最佳配置)
print("\n【5. 配置SGDClassifier】")
print("使用最终优化报告中的最佳配置:")
print("  loss='squared_hinge'")
print("  penalty='l1'")
print("  alpha=0.0038")
print("  eta0=0.005")
print("  learning_rate='optimal'")
print("  max_iter=5000")
print("  class_weight=None")

sgd = SGDClassifier(
    loss='squared_hinge',      # 平方合页损失
    penalty='l1',              # L1正则化
    alpha=0.0038,              # 学习率倒数 (正则化强度)
    eta0=0.005,                # 初始学习率
    learning_rate='optimal',   # 学习率调度策略
    max_iter=5000,             # 最大迭代次数
    class_weight=None,          # 不使用类别权重
    random_state=42
)

# 6. 8折分层交叉验证
print("\n【6. 8折分层交叉验证】")
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

print(f"交叉验证策略: 8折分层交叉验证")
print(f"样本分布: 类别1={sum(y==1)}, 类别2={sum(y==2)}")

# 7. 执行交叉验证
print("\n【7. 执行交叉验证】")
scores = cross_val_score(sgd, X_scaled, y, cv=skf, scoring='accuracy')

print(f"\n各折准确率:")
for i, score in enumerate(scores, 1):
    print(f"  折{i}: {score*100:.2f}%")

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"\n平均准确率: {mean_accuracy*100:.2f}%")
print(f"标准差: ±{std_accuracy*100:.2f}%")
print(f"准确率范围: [{(mean_accuracy - std_accuracy)*100:.2f}%, {(mean_accuracy + std_accuracy)*100:.2f}%]")

# 8. 结果对比
print("\n" + "="*80)
print("复现结果")
print("="*80)

print(f"\n最终优化报告中的准确率: 92.86%")
print(f"本次复现的准确率: {mean_accuracy*100:.2f}%")

if abs(mean_accuracy - 0.9286) < 0.01:
    print("\n[成功] 成功复现92.86%的准确率！")
    print(f"误差: {abs(mean_accuracy - 0.9286)*100:.2f}%")
else:
    print(f"\n[警告] 复现准确率与报告有差异")
    print(f"差异: {abs(mean_accuracy - 0.9286)*100:.2f}%")
    if mean_accuracy < 0.9286:
        print(f"复现准确率较低: {(0.9286 - mean_accuracy)*100:.2f}%")
    else:
        print(f"复现准确率更高: {(mean_accuracy - 0.9286)*100:.2f}%")

# 9. 训练完整模型并保存配置信息
print("\n【8. 训练完整模型】")
sgd.fit(X_scaled, y)

print(f"\n模型训练完成")
print(f"收敛迭代次数: {sgd.n_iter_}")

# 获取模型系数（squared_hinge损失函数支持coef_）
if hasattr(sgd, 'coef_'):
    coefficients = sgd.coef_[0]
    intercept = sgd.intercept_[0]

    print(f"\n决策边界:")
    print(f"  截距: {intercept:.4f}")
    print(f"\n特征系数:")
    for i, (feat, coef) in enumerate(zip(best_features, coefficients), 1):
        print(f"  {i}. {feat}: {coef:.4f}")

    # 特征重要性（按系数绝对值排序）
    feature_importance = pd.DataFrame({
        'Feature': best_features,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)

    print(f"\n特征重要性排序:")
    for i, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Abs_Coefficient']:.4f}")
else:
    print("\n当前损失函数不支持输出系数")

# 10. 输出完整配置
print("\n" + "="*80)
print("完整配置信息")
print("="*80)

print(f"\n数据集:")
print(f"  文件: CARCT1_E1_FG2_processed_translated.csv")
print(f"  样本数: {len(data)}")
print(f"  特征数: {len(best_features)}")
print(f"  类别1: {sum(y==1)}")
print(f"  类别2: {sum(y==2)}")

print(f"\n特征选择:")
print(f"  方法: 手动选择（基于RFE结果）")
print(f"  选择特征数: {len(best_features)}")
print(f"  特征列表: {', '.join(best_features)}")

print(f"\n标准化:")
print(f"  方法: RobustScaler")
print(f"  说明: 基于中位数和四分位数,对异常值鲁棒")

print(f"\n模型:")
print(f"  类型: SGDClassifier")
print(f"  loss: squared_hinge")
print(f"  penalty: l1")
print(f"  alpha: 0.0038")
print(f"  eta0: 0.005")
print(f"  learning_rate: optimal")
print(f"  max_iter: 5000")
print(f"  class_weight: None")
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

# 11. 预测示例
print("\n【9. 预测示例】")
sample_idx = 0
sample_features = X_selected.iloc[sample_idx:sample_idx+1]
sample_scaled = scaler.transform(sample_features)
prediction = sgd.predict(sample_scaled)
probability = sgd.decision_function(sample_scaled)[0] if hasattr(sgd, 'decision_function') else None

print(f"\n样本 {sample_idx + 1} (ID: {data.iloc[sample_idx]['ID']}):")
print(f"  真实标签: FCR={y.iloc[sample_idx]}")
print(f"  预测标签: FCR={prediction[0]}")
if probability is not None:
    print(f"  决策函数值: {probability:.4f}")
print(f"  样本特征: {dict(sample_features.iloc[0])}")

# 12. 保存模型和信息
print("\n【10. 保存模型和信息】")
base_filename = '92.86%准确率SGD模型_E1数据集'
joblib_filename = base_filename + '.joblib'
txt_filename = base_filename + '_信息.txt'

# 保存完整模型信息
model_info = {
    'model': sgd,
    'scaler': scaler,
    'optimal_features': best_features,
    'feature_importance': feature_importance if hasattr(sgd, 'coef_') else None,
    'mean_accuracy': mean_accuracy,
    'std_accuracy': std_accuracy,
    'cv_scores': scores,
    'coefficients': coefficients if hasattr(sgd, 'coef_') else None,
    'intercept': intercept if hasattr(sgd, 'coef_') else None
}

joblib.dump(model_info, joblib_filename)
print(f"模型已保存: {joblib_filename}")

# 保存模型信息到txt文件
with open(txt_filename, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("92.86%准确率SGD模型 - E1数据集\n")
    f.write("="*80 + "\n\n")
    
    f.write("【数据集信息】\n")
    f.write(f"  文件: CARCT1_E1_FG2_processed_translated.csv\n")
    f.write(f"  样本数: {len(data)}\n")
    f.write(f"  特征数: {len(best_features)}\n")
    f.write(f"  类别1: {sum(y==1)}\n")
    f.write(f"  类别2: {sum(y==2)}\n\n")
    
    f.write("【特征选择】\n")
    f.write(f"  选择特征数: {len(best_features)}\n")
    f.write(f"  特征列表: {', '.join(best_features)}\n\n")
    
    f.write("【模型配置】\n")
    f.write("  模型类型: SGDClassifier\n")
    f.write(f"  loss: squared_hinge\n")
    f.write(f"  penalty: l1\n")
    f.write(f"  alpha: 0.0038\n")
    f.write(f"  eta0: 0.005\n")
    f.write(f"  learning_rate: optimal\n")
    f.write(f"  max_iter: 5000\n")
    f.write(f"  class_weight: None\n")
    f.write(f"  random_state: 42\n\n")
    
    f.write("【标准化】\n")
    f.write("  方法: RobustScaler\n")
    f.write("  说明: 基于中位数和四分位数,对异常值鲁棒\n\n")
    
    f.write("【交叉验证结果】\n")
    f.write(f"  8折交叉验证平均准确率: {mean_accuracy*100:.2f}%\n")
    f.write(f"  标准差: ±{std_accuracy*100:.2f}%\n")
    f.write(f"  各折准确率:\n")
    for i, score in enumerate(scores, 1):
        f.write(f"    折{i}: {score*100:.2f}%\n")
    f.write("\n")
    
    if hasattr(sgd, 'coef_'):
        f.write("【模型参数】\n")
        f.write(f"  截距: {intercept:.4f}\n")
        f.write(f"  收敛迭代次数: {sgd.n_iter_}\n\n")
        
        f.write("【特征系数】\n")
        for i, (feat, coef) in enumerate(zip(best_features, coefficients), 1):
            f.write(f"  {i}. {feat}: {coef:.4f}\n")
        f.write("\n")
        
        f.write("【特征重要性排序】\n")
        for i, row in feature_importance.iterrows():
            f.write(f"  {i}. {row['Feature']}: {row['Abs_Coefficient']:.4f}\n")
        f.write("\n")
    
    f.write("【使用说明】\n")
    f.write("1. 加载模型:\n")
    f.write(f"   model_info = joblib.load('{joblib_filename}')\n")
    f.write(f"   model = model_info['model']\n")
    f.write(f"   scaler = model_info['scaler']\n")
    f.write(f"   features = model_info['optimal_features']\n\n")
    f.write("2. 预测新数据:\n")
    f.write("   - 确保新数据包含所有最优特征\n")
    f.write("   - 使用scaler.transform()标准化\n")
    f.write("   - 使用model.predict()预测\n\n")
    f.write("="*80 + "\n")

print(f"模型信息已保存: {txt_filename}")
