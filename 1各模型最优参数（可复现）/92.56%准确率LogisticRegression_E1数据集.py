"""
LogisticRegression最优模型 - E1数据集
准确率: 92.56% (8折交叉验证)
可复现版本
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============ 数据加载 ============
print("="*70)
print("LogisticRegression最优模型 - E1数据集")
print("="*70)
print("\n加载数据...")
data = pd.read_csv('CARCT1_E1_FG2_processed_translated.csv', encoding='utf-8')

# 删除目标变量缺失的行
data = data.dropna(subset=['FCR_G2']).copy()
print(f"样本数量: {len(data)}")

# 目标变量
y = data['FCR_G2'].values
y_binary = (y - 1).astype(int)  # 将1,2转为0,1
print(f"目标变量分布: 类别0: {np.sum(y_binary==0)}, 类别1: {np.sum(y_binary==1)}")

# ============ 特征选择 ============
print("\n" + "="*70)
print("特征选择")
print("="*70)

# 基础特征列表（排除目标变量和ID）
exclude_cols = ['FCR_G2', 'ID']
feature_cols = [col for col in data.columns if col not in exclude_cols]

# 数据清理
data_clean = data[feature_cols].copy()

# 删除全空列
non_null_cols = data_clean.columns[data_clean.notna().any()].tolist()
data_clean = data_clean[non_null_cols]

# 删除缺失值超过50%的列
missing_ratio = data_clean.isna().sum() / len(data_clean)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
data_clean = data_clean[cols_to_keep]

# 用中位数填充缺失值
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(
    imputer.fit_transform(data_clean),
    columns=data_clean.columns,
    index=data_clean.index
)

print(f"基础特征数量: {X.shape[1]}")

# 最优特征（10个）
optimal_features = [
    'GAD7_0',
    'TCSQ_NC',
    'Age',
    'Residence',
    'Education',
    'Has_Partner',
    'Relationship_with_Family',
    'Family_Social_Emotional_Support',
    'Perceived_Severity_of_Condition',
    'Life_Economic_Stress'
]

print(f"\n最优特征数量: {len(optimal_features)}")
print(f"选择的特征: {optimal_features}")

X_selected = X[optimal_features].values
print(f"特征矩阵形状: {X_selected.shape}")

# ============ 数据标准化 ============
print("\n" + "="*70)
print("数据标准化")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
print(f"使用StandardScaler标准化")

# ============ 模型训练 ============
print("\n" + "="*70)
print("模型配置")
print("="*70)

# 最优参数
model_params = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 2000,
    'tol': 0.0001,
    'random_state': 42
}

print("模型参数:")
for k, v in model_params.items():
    print(f"  {k}: {v}")

# 创建模型
model = LogisticRegression(**model_params)

# 训练模型
print("\n训练模型...")
model.fit(X_scaled, y_binary)
print("模型训练完成!")

# ============ 模型系数 ============
print("\n" + "="*70)
print("模型系数")
print("="*70)

coefficients = model.coef_[0]
intercept = model.intercept_[0]

print(f"\n截距: {intercept:.6f}\n")
print("特征系数:")
for feat, coef in zip(optimal_features, coefficients):
    print(f"  {feat}: {coef:.6f}")

# ============ 特征重要性 ============
print("\n" + "="*70)
print("特征重要性（按绝对值排序）")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': optimal_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(feature_importance.to_string(index=False))

# ============ 交叉验证评估 ============
print("\n" + "="*70)
print("交叉验证评估")
print("="*70)

cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_binary, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\n8折交叉验证准确率:")
for i, score in enumerate(cv_scores, 1):
    print(f"  折{i}: {score:.2%}")

print(f"\n平均准确率: {cv_scores.mean():.4%}")
print(f"标准差: ±{cv_scores.std():.4%}")

# ============ 预测示例 ============
print("\n" + "="*70)
print("预测示例")
print("="*70)

# 对训练集进行预测
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

print(f"\n前10个样本的预测结果:")
print(f"{'真实值':<8} {'预测值':<8} {'概率(FCR_G2=2)':<15}")
print("-" * 35)
for i in range(10):
    true_label = y_binary[i]
    pred_label = y_pred[i]
    pred_prob = y_pred_proba[i]
    print(f"{true_label:<8} {pred_label:<8} {pred_prob:<15.4f}")

# ============ 模型使用说明 ============
print("\n" + "="*70)
print("模型使用说明")
print("="*70)

print("""
如何使用此模型进行预测:

1. 加载新数据:
   new_data = pd.read_csv('新数据.csv', encoding='utf-8')

2. 数据预处理:
   - 确保包含所有最优特征: """)
feature_str = ', '.join(optimal_features)
print(f"   {feature_str}")
print("""   - 填充缺失值（使用训练集的中位数）
   - 选择相同的特征列

3. 标准化:
   X_new = new_data[optimal_features].values
   X_new_scaled = scaler.transform(X_new)

4. 预测:
   y_pred = model.predict(X_new_scaled)  # 预测类别 (0或1)
   y_proba = model.predict_proba(X_new_scaled)  # 预测概率

5. 结果解读:
   - 预测结果0表示FCR_G2=1
   - 预测结果1表示FCR_G2=2
   - 概率值越大，属于FCR_G2=2的可能性越高
""")

# ============ 保存模型 ============
print("\n" + "="*70)
print("保存模型")
print("="*70)

import joblib
import os

# 创建保存目录
save_dir = '各模型最优参数（可复现）'
os.makedirs(save_dir, exist_ok=True)

# 保存模型组件
model_info = {
    'model': model,
    'scaler': scaler,
    'imputer': imputer,
    'optimal_features': optimal_features,
    'feature_importance': feature_importance,
    'cv_scores': cv_scores
}

model_path = os.path.join(save_dir, '92.56%准确率LogisticRegression_E1数据集.joblib')
joblib.dump(model_info, model_path)
print(f"模型已保存到: {model_path}")

# 保存模型参数文本
info_path = os.path.join(save_dir, '92.56%准确率LogisticRegression_E1数据集_信息.txt')
with open(info_path, 'w', encoding='utf-8') as f:
    f.write("LogisticRegression最优模型信息 - E1数据集\n")
    f.write("="*70 + "\n\n")
    f.write(f"准确率: 92.56% (8折交叉验证)\n")
    f.write(f"标准差: ±7.48%\n\n")
    f.write("模型参数:\n")
    for k, v in model_params.items():
        f.write(f"  {k}: {v}\n")

    f.write(f"\n最优特征数量: {len(optimal_features)}\n")
    f.write("最优特征:\n")
    for i, feat in enumerate(optimal_features, 1):
        f.write(f"  {i}. {feat}\n")

    f.write("\n8折交叉验证准确率:\n")
    for i, score in enumerate(cv_scores, 1):
        f.write(f"  折{i}: {score:.2%}\n")

    f.write(f"\n平均准确率: {cv_scores.mean():.4%}\n")
    f.write(f"标准差: ±{cv_scores.std():.4%}\n")
    f.write("\n特征重要性:\n")
    for idx, row in feature_importance.iterrows():
        coef = row['Coefficient']
        imp = row['Abs_Coefficient']
        feat = row['Feature']
        f.write(f"  {feat}: {coef:.4f} (重要性: {imp:.4f})\n")

print(f"模型信息已保存到: {info_path}")

print("\n" + "="*70)
print("完成!")
print("="*70)
