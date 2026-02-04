"""
LogisticRegression模型调参 - E1数据集 - 最终优化（目标92%准确率）
基于90.77%的配置进行精细调优
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform, uniform
import warnings
warnings.filterwarnings('ignore')

# ============ 数据加载 ============
print("="*70)
print("LogisticRegression模型调参 - E1数据集（最终优化）")
print("="*70)
print("\n加载数据...")
data = pd.read_csv('CARCT1_E1_FG2_processed_translated.csv', encoding='utf-8')

# 删除目标变量缺失的行
data = data.dropna(subset=['FCR_G2']).copy()
print(f"删除目标变量缺失值后，剩余样本数: {len(data)}")

# 目标变量
y = data['FCR_G2']
print(f"目标变量分布:\n{y.value_counts()}")

# ============ 特征工程 ============
print("\n" + "="*70)
print("特征工程（简化版）")
print("="*70)

# 获取所有数值特征（排除目标变量和ID）
exclude_cols = ['FCR_G2', 'ID']
feature_cols = [col for col in data.columns if col not in exclude_cols]

# 删除全空列
data_clean = data[feature_cols].copy()
non_null_cols = data_clean.columns[data_clean.notna().any()].tolist()
data_clean = data_clean[non_null_cols]
print(f"删除全空列后，剩余 {len(non_null_cols)} 个特征")

# 删除缺失值超过50%的列
missing_ratio = data_clean.isna().sum() / len(data_clean)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
data_clean = data_clean[cols_to_keep]
print(f"删除缺失值>50%的列后，剩余 {len(cols_to_keep)} 个特征")

# 使用SimpleImputer填充缺失值
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(
    imputer.fit_transform(data_clean),
    columns=data_clean.columns,
    index=data_clean.index
)
print(f"用中位数填充后，共 {X.shape[1]} 个特征")

# ============ 第一阶段：尝试不同特征数量 ============
print("\n" + "="*70)
print("第一阶段：测试不同特征数量")
print("="*70)

scaler = StandardScaler()

best_score_overall = 0
best_n_features = 0
best_selected = None
best_params_overall = None

for n_features in range(5, 16):
    print(f"\n测试 {n_features} 个特征...")

    # RFE特征选择
    X_scaled = scaler.fit_transform(X)
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(
        estimator=lr_base,
        n_features_to_select=n_features,
        step=1
    )
    rfe.fit(X_scaled, y)

    selected_features = X.columns[rfe.support_].tolist()
    X_selected = X_scaled[:, rfe.support_]
    print(f"  选择的特征: {selected_features}")

    # 网格搜索调参（基于90.77%的最佳参数周围）
    param_grid = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
        'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear'],
        'class_weight': [None, 'balanced', {0: 1, 1: 1.2}, {0: 1, 1: 1.5}],
        'max_iter': [2000, 3000, 5000]
    }

    lr = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_selected, y)
    print(f"  最佳准确率: {grid_search.best_score_:.4%}")
    print(f"  最佳参数: C={grid_search.best_params_['C']}, penalty={grid_search.best_params_['penalty']}")

    if grid_search.best_score_ > best_score_overall:
        best_score_overall = grid_search.best_score_
        best_n_features = n_features
        best_selected = selected_features
        best_params_overall = grid_search.best_params_
        best_mask = rfe.support_
        print(f"  ★ 新的最佳配置! ({best_n_features}个特征, 准确率={best_score_overall:.4%})")

print(f"\n第一阶段最佳: {best_n_features}个特征, 准确率={best_score_overall:.4%}")

# ============ 第二阶段：基于最佳配置的精细化调优 ============
print("\n" + "="*70)
print("第二阶段：精细化调优")
print("="*70)

# 使用RFE的mask来选择特征
X_scaled_final = scaler.fit_transform(X)
X_selected_final = X_scaled_final[:, best_mask]

# 在最佳参数周围更精细地搜索
best_C = best_params_overall['C']
best_penalty = best_params_overall['penalty']
best_solver = best_params_overall['solver']

param_grid_fine = {
    'C': [best_C * 0.5, best_C * 0.7, best_C * 0.8, best_C * 0.9,
          best_C, best_C * 1.1, best_C * 1.2, best_C * 1.3, best_C * 1.5],
    'penalty': [best_penalty],
    'solver': [best_solver],
    'class_weight': [best_params_overall['class_weight']],
    'max_iter': [best_params_overall['max_iter'], 5000],
    'tol': [1e-4, 1e-5]
}

lr = LogisticRegression(random_state=42)
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

grid_search_fine = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_fine,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_fine.fit(X_selected_final, y)

print(f"\n精细化搜索最佳准确率: {grid_search_fine.best_score_:.4%}")
print(f"最佳参数: {grid_search_fine.best_params_}")

# ============ 第三阶段：RandomizedSearchCV探索 ============
print("\n" + "="*70)
print("第三阶段：RandomizedSearchCV探索")
print("="*70)

param_distributions = {
    'C': loguniform(0.01, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear', 'lbfgs'],
    'class_weight': [None, 'balanced', {0: 1, 1: 1.1}, {0: 1, 1: 1.2}, {0: 1, 1: 1.3}, {0: 1, 1: 1.5}],
    'max_iter': [2000, 3000, 4000, 5000, 8000],
    'tol': [1e-3, 1e-4, 1e-5, 1e-6]
}

random_search = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_distributions=param_distributions,
    n_iter=200,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_selected_final, y)

print(f"\nRandomizedSearchCV最佳准确率: {random_search.best_score_:.4%}")
print(f"最佳参数: {random_search.best_params_}")

# ============ 第四阶段：尝试不同标准化方法 ============
print("\n" + "="*70)
print("第四阶段：对比不同标准化方法")
print("="*70)

scalers_to_test = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler()
}

results = {}
for scaler_name, scaler_obj in scalers_to_test.items():
    print(f"\n测试 {scaler_name}...")
    X_scaled = scaler_obj.fit_transform(X)
    X_selected = X_scaled[:, best_mask]

    cv_scores = cross_val_score(
        LogisticRegression(**grid_search_fine.best_params_, random_state=42),
        X_selected, y,
        cv=StratifiedKFold(n_splits=8, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )

    results[scaler_name] = cv_scores.mean()
    print(f"  平均准确率: {cv_scores.mean():.4%} ± {cv_scores.std():.4%}")

best_scaler = max(results.keys(), key=lambda k: results[k])
print(f"\n最佳标准化方法: {best_scaler} ({results[best_scaler]:.4%})")

# ============ 选择最终最佳模型 ============
print("\n" + "="*70)
print("最终结果")
print("="*70)

all_results = {
    '精细网格搜索': (grid_search_fine.best_score_, grid_search_fine.best_params_, grid_search_fine.best_estimator_),
    '随机搜索': (random_search.best_score_, random_search.best_params_, random_search.best_estimator_),
    f'{best_scaler}': (results[best_scaler], grid_search_fine.best_params_, grid_search_fine.best_estimator_)
}

best_method = max(all_results.keys(), key=lambda k: all_results[k][0])
best_score, best_params, best_model = all_results[best_method]

print(f"\n{best_method} 表现最佳!")
print(f"\n最佳准确率: {best_score:.4%}")
print(f"最佳参数:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ============ 交叉验证详细结果 ============
print("\n" + "="*70)
print("8折交叉验证详细结果")
print("="*70)

scaler_final = StandardScaler() if best_scaler == 'StandardScaler' else RobustScaler()
X_scaled = scaler_final.fit_transform(X)
X_selected = X_scaled[:, best_mask]

cv_scores = cross_val_score(best_model, X_selected, y,
                           cv=StratifiedKFold(n_splits=8, shuffle=True, random_state=42),
                           scoring='accuracy', n_jobs=-1)
print(f"各折准确率: {[f'{score:.2%}' for score in cv_scores]}")
print(f"平均准确率: {cv_scores.mean():.4%} ± {cv_scores.std():.4%}")

# ============ 模型特征重要性 ============
print("\n" + "="*70)
print("模型特征重要性")
print("="*70)

best_model.fit(X_selected, y)
if hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': best_selected,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\n特征系数（按绝对值排序）:")
    print(feature_importance.to_string(index=False))

# ============ 保存结果 ============
print("\n" + "="*70)
print("保存结果")
print("="*70)

result_text = f"""
LogisticRegression模型调参结果 - E1数据集（最终优化）
{'='*70}

最佳准确率: {best_score:.4%}
最佳方法: {best_method}
最佳标准化: {best_scaler}
特征数量: {len(best_selected)}

最佳参数:
"""
for k, v in best_params.items():
    result_text += f"  {k}: {v}\n"

result_text += f"""
选择的特征:
"""
for i, feat in enumerate(best_selected, 1):
    result_text += f"  {i}. {feat}\n"

result_text += f"""
8折交叉验证详细结果:
"""
for i, score in enumerate(cv_scores, 1):
    result_text += f"  折{i}: {score:.2%}\n"

result_text += f"""
平均准确率: {cv_scores.mean():.4%} ± {cv_scores.std():.4%}

特征重要性:
"""
if hasattr(best_model, 'coef_'):
    for idx, row in feature_importance.iterrows():
        result_text += f"  {row['Feature']}: {row['Coefficient']:.4f} (重要性: {row['Abs_Coefficient']:.4f})\n"

# 保存到文件
with open('LogisticRegression调参结果_E1数据集_最终优化.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)

print("结果已保存到: LogisticRegression调参结果_E1数据集_最终优化.txt")

print("\n" + "="*70)
print("调参完成!")
print("="*70)

# 检查是否达到目标
if cv_scores.mean() >= 0.92:
    print(f"\n[OK] 成功达到92%目标! (实际: {cv_scores.mean():.4%})")
else:
    gap = 0.92 - cv_scores.mean()
    print(f"\n未达到92%目标, 差距: {gap:.4%}")
