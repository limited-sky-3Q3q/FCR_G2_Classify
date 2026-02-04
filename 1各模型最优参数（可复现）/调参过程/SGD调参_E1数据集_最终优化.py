# -*- coding: utf-8 -*-
"""
基于CARCT1_E1_FG2_processed_translated.csv的线性模型调参（最终优化版）
基于RandomizedSearchCV发现的92.86%配置进行优化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("线性模型调参（最终优化版）- 目准确率92%+")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
data = pd.read_csv('CARCT1_E1_FG2_processed_translated.csv', encoding='utf-8')
data = data.dropna(subset=['FCR_G2', 'ID'])

exclude_cols = ['ID', 'FCR_G2']
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()
y = data['FCR_G2'].copy().astype(int)

print(f"样本数: {len(data)}")
print(f"特征数: {len(feature_cols)}")

# 清理数据
X = X.dropna(axis=1, how='all')
missing_ratio = X.isnull().sum() / len(X)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
X = X[cols_to_keep]
X = X.fillna(X.median())
print(f"清理后特征数: {len(X.columns)}")

# 2. 使用最佳特征选择（RFE, 10个特征）
print("\n【2. 特征选择】")
best_features = ['GAD7_0', 'Residence', 'Marriage', 'Education', 'Partner_Monthly_Income',
                 'Relationship_with_Family', 'Family_Social_Emotional_Support', 'Chemotherapy',
                 'Perceived_Severity_of_Condition', 'Duration_Aware_of_Cancer_Diagnosis']
X_selected = X[best_features]
print(f"使用10个特征: {', '.join(best_features)}")

# 3. 标准化
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

# 4. 基于92.86%配置进行精细优化
print("\n【3. 精细参数搜索】")
print("基于RandomizedSearchCV发现的最佳配置:")
print("  loss: squared_hinge")
print("  penalty: l1")
print("  alpha: 0.004")
print("  max_iter: 5000")
print("  class_weight: None")
print("  eta0: 0.01")
print("  learning_rate: optimal")

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# 在alpha=0.004周围进行超精细搜索
param_grid = {
    'loss': ['squared_hinge', 'log_loss'],
    'penalty': ['l1', 'l2'],
    'alpha': [0.0035, 0.0038, 0.004, 0.0042, 0.0045, 0.005],
    'max_iter': [5000],
    'class_weight': [None, 'balanced'],
    'random_state': [42],
    'eta0': [0.005, 0.008, 0.01, 0.012, 0.015],
    'learning_rate': ['optimal', 'invscaling']
}

grid_search = GridSearchCV(
    SGDClassifier(),
    param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\n执行精细网格搜索...")
grid_search.fit(X_scaled, y)

print(f"\n最佳参数:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"最佳准确率: {grid_search.best_score_*100:.2f}%")

# 5. 如果还没到92%，尝试更大的alpha范围
if grid_search.best_score_ < 0.92:
    print("\n【4. 扩大alpha搜索范围】")
    best_alpha = grid_search.best_params_['alpha']

    # 更大的alpha搜索范围
    alpha_range = np.linspace(0.002, 0.008, 31)
    alpha_scores = []

    print(f"在{best_alpha}周围扩大搜索...")
    for alpha in alpha_range:
        params = grid_search.best_params_.copy()
        params['alpha'] = alpha

        sgd = SGDClassifier(**params)
        scores = cross_val_score(sgd, X_scaled, y, cv=skf, scoring='accuracy')

        alpha_scores.append({
            'alpha': alpha,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        })

        if scores.mean() > grid_search.best_score_:
            print(f"  更新最佳: alpha={alpha:.6f}, {scores.mean()*100:.2f}%")

    best_alpha_result = max(alpha_scores, key=lambda x: x['mean_accuracy'])
    print(f"\n最佳alpha: {best_alpha_result['alpha']:.6f}")
    print(f"准确率: {best_alpha_result['mean_accuracy']*100:.2f}%")

    # 更新最佳参数
    grid_search.best_params_['alpha'] = best_alpha_result['alpha']
    grid_search.best_score_ = best_alpha_result['mean_accuracy']

# 6. 尝试不同的max_iter
print("\n【5. 优化迭代次数】")
for max_iter in [3000, 5000, 8000, 10000]:
    params = grid_search.best_params_.copy()
    params['max_iter'] = max_iter

    sgd = SGDClassifier(**params)
    scores = cross_val_score(sgd, X_scaled, y, cv=skf, scoring='accuracy')

    print(f"  max_iter={max_iter}: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    if scores.mean() > grid_search.best_score_:
        grid_search.best_params_['max_iter'] = max_iter
        grid_search.best_score_ = scores.mean()

# 7. 尝试调整class_weight
print("\n【6. 优化类别权重】")
class_weight_options = [None, 'balanced', {1: 1.2, 2: 0.8}, {1: 1.5, 2: 0.7}]

for cw in class_weight_options:
    params = grid_search.best_params_.copy()
    params['class_weight'] = cw

    sgd = SGDClassifier(**params)
    scores = cross_val_score(sgd, X_scaled, y, cv=skf, scoring='accuracy')

    cw_str = str(cw) if cw is not None else 'None'
    print(f"  class_weight={cw_str}: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    if scores.mean() > grid_search.best_score_:
        grid_search.best_params_['class_weight'] = cw
        grid_search.best_score_ = scores.mean()

# 8. 最终评估
print("\n【7. 最终模型评估】")
final_params = grid_search.best_params_
best_model = SGDClassifier(**final_params)

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X_scaled, y, cv=skf, scoring='accuracy')

print(f"\n各折准确率:")
for i, score in enumerate(scores, 1):
    print(f"  折{i}: {score*100:.2f}%")

mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"\n平均准确率: {mean_accuracy*100:.2f}%")
print(f"标准差: ±{std_accuracy*100:.2f}%")

# 训练完整模型
print("\n【8. 训练完整模型】")
best_model.fit(X_scaled, y)

print(f"模型训练完成")
print(f"收敛迭代次数: {best_model.n_iter_}")

# 获取模型系数（如果支持）
if hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    intercept = best_model.intercept_[0]

    print(f"\n决策边界:")
    print(f"  截距: {intercept:.4f}")
    print(f"\n特征系数:")
    for i, (feat, coef) in enumerate(zip(best_features, coefficients), 1):
        print(f"  {i}. {feat}: {coef:.4f}")

    # 特征重要性排序
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

# 9. 总结
print("\n" + "="*80)
print("最终配置总结")
print("="*80)

print(f"\n数据集: CARCT1_E1_FG2_processed_translated.csv")
print(f"  样本数: {len(data)}")
print(f"  特征数: {len(best_features)}")
print(f"  类别1: {sum(y==1)}")
print(f"  类别2: {sum(y==2)}")

print(f"\n特征列表:")
for i, feat in enumerate(best_features, 1):
    print(f"  {i}. {feat}")

print(f"\n标准化: RobustScaler")

print(f"\n最佳参数:")
for param, value in final_params.items():
    print(f"  {param}: {value}")

print(f"\n性能:")
print(f"  平均准确率: {mean_accuracy*100:.2f}%")
print(f"  标准差: ±{std_accuracy*100:.2f}%")

if mean_accuracy >= 0.92:
    print("\n[成功] 达到92%以上的准确率目标！")
elif mean_accuracy >= 0.91:
    print(f"\n[优秀] 准确率{mean_accuracy*100:.2f}%，距离92%目标还差{(0.92 - mean_accuracy)*100:.2f}%")
else:
    print(f"\n[提示] 当前准确率{mean_accuracy*100:.2f}%，距离92%目标还差{(0.92 - mean_accuracy)*100:.2f}%")

print("\n" + "="*80)
print("调参完成")
print("="*80)

# 保存结果
output_file = '调参结果_E1数据集_最终优化.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("线性模型调参结果（最终优化版）- CARCT1_E1_FG2_processed_translated.csv\n")
    f.write("="*80 + "\n\n")

    f.write(f"数据集:\n")
    f.write(f"  样本数: {len(data)}\n")
    f.write(f"  特征数: {len(best_features)}\n")
    f.write(f"  类别1: {sum(y==1)}\n")
    f.write(f"  类别2: {sum(y==2)}\n\n")

    f.write(f"特征列表:\n")
    for i, feat in enumerate(best_features, 1):
        f.write(f"  {i}. {feat}\n\n")

    f.write(f"标准化: RobustScaler\n\n")

    f.write(f"最佳参数:\n")
    for param, value in final_params.items():
        f.write(f"  {param}: {value}\n\n")

    f.write(f"性能:\n")
    f.write(f"  平均准确率: {mean_accuracy*100:.2f}%\n")
    f.write(f"  标准差: ±{std_accuracy*100:.2f}%\n\n")

    f.write(f"各折准确率:\n")
    for i, score in enumerate(scores, 1):
        f.write(f"  折{i}: {score*100:.2f}%\n")

    if hasattr(best_model, 'coef_'):
        f.write(f"\n特征系数:\n")
        for i, (feat, coef) in enumerate(zip(best_features, coefficients), 1):
            f.write(f"  {i}. {feat}: {coef:.4f}\n")

print(f"\n结果已保存到: {output_file}")
