"""
批量调参 - E1数据集
对6个模型进行调参：ExtraTrees、RandomForest、GradientBoosting、SVM、KNN、NeuralNetwork
基于CARCT1_E1_FG2_processed_translated.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, uniform
import warnings
warnings.filterwarnings('ignore')

# ============ 数据加载 ============
print("="*70)
print("批量调参 - E1数据集")
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

# ============ 数据清理 ============
print("\n" + "="*70)
print("数据清理")
print("="*70)

exclude_cols = ['FCR_G2', 'ID']
feature_cols = [col for col in data.columns if col not in exclude_cols]

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

print(f"特征数量: {X.shape[1]}")

# ============ 特征选择 ============
print("\n" + "="*70)
print("特征选择（使用LogisticRegression RFE）")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_base = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=lr_base, n_features_to_select=10, step=1)
rfe.fit(X_scaled, y_binary)

selected_features = X.columns[rfe.support_].tolist()
X_selected = X_scaled[:, rfe.support_]

print(f"选择的特征: {selected_features}")

# ============ 定义模型调参函数 ============
def tune_model(model_name, model, param_grid, X, y, cv):
    print(f"\n{'='*70}")
    print(f"调参: {model_name}")
    print('='*70)

    # 网格搜索
    if len(list(param_grid.values())[0]) > 30:  # 参数空间大时使用随机搜索
        print("使用RandomizedSearchCV...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=200,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        print("使用GridSearchCV...")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

    search.fit(X, y)

    print(f"\n{model_name}最佳准确率: {search.best_score_:.4%}")
    print(f"最佳参数: {search.best_params_}")

    # 交叉验证详细结果
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(search.best_estimator_, X, y,
                               cv=StratifiedKFold(n_splits=8, shuffle=True, random_state=42),
                               scoring='accuracy', n_jobs=-1)
    print(f"8折交叉验证准确率: {[f'{s:.2%}' for s in cv_scores]}")
    print(f"平均准确率: {cv_scores.mean():.4%} ± {cv_scores.std():.4%}")

    return search.best_score_, search.best_params_, search.best_estimator_

# ============ 模型调参配置 ============
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

# 1. ExtraTrees
print("\n" + "#"*70)
print("# 1/6: ExtraTreesClassifier")
print("#"*70)

et_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42]
}

et_score, et_params, et_model = tune_model(
    'ExtraTreesClassifier',
    ExtraTreesClassifier(),
    et_param_grid,
    X_selected, y_binary, cv
)

# 2. RandomForest
print("\n" + "#"*70)
print("# 2/6: RandomForestClassifier")
print("#"*70)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
}

rf_score, rf_params, rf_model = tune_model(
    'RandomForestClassifier',
    RandomForestClassifier(),
    rf_param_grid,
    X_selected, y_binary, cv
)

# 3. GradientBoosting
print("\n" + "#"*70)
print("# 3/6: GradientBoostingClassifier")
print("#"*70)

gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0],
    'random_state': [42]
}

gb_score, gb_params, gb_model = tune_model(
    'GradientBoostingClassifier',
    GradientBoostingClassifier(),
    gb_param_grid,
    X_selected, y_binary, cv
)

# 4. SVM
print("\n" + "#"*70)
print("# 4/6: SVC")
print("#"*70)

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
}

svm_score, svm_params, svm_model = tune_model(
    'SVC',
    SVC(),
    svm_param_grid,
    X_selected, y_binary, cv
)

# 5. KNN
print("\n" + "#"*70)
print("# 5/6: KNeighborsClassifier")
print("#"*70)

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_score, knn_params, knn_model = tune_model(
    'KNeighborsClassifier',
    KNeighborsClassifier(),
    knn_param_grid,
    X_selected, y_binary, cv
)

# 6. Neural Network (MLP)
print("\n" + "#"*70)
print("# 6/6: MLPClassifier")
print("#"*70)

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [1000, 2000],
    'random_state': [42]
}

mlp_score, mlp_params, mlp_model = tune_model(
    'MLPClassifier',
    MLPClassifier(),
    mlp_param_grid,
    X_selected, y_binary, cv
)

# ============ 结果总结 ============
print("\n" + "="*70)
print("调参结果总结")
print("="*70)

results = {
    'ExtraTrees': (et_score, et_params, et_model),
    'RandomForest': (rf_score, rf_params, rf_model),
    'GradientBoosting': (gb_score, gb_params, gb_model),
    'SVM': (svm_score, svm_params, svm_model),
    'KNN': (knn_score, knn_params, knn_model),
    'NeuralNetwork': (mlp_score, mlp_params, mlp_model)
}

sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

print(f"\n{'模型':<20} {'准确率':<10} {'参数':<50}")
print("-" * 80)
for name, (score, params, _) in sorted_results:
    params_str = str(params)[:50] + '...' if len(str(params)) > 50 else str(params)
    print(f"{name:<20} {score:>8.4%}  {params_str:<50}")

# 保存调参结果
with open('批量调参结果_E1数据集.txt', 'w', encoding='utf-8') as f:
    f.write("批量调参结果 - E1数据集\n")
    f.write("="*70 + "\n\n")
    f.write(f"数据集: CARCT1_E1_FG2_processed_translated.csv\n")
    f.write(f"样本数: {len(data)}\n")
    f.write(f"特征数: {len(selected_features)}\n")
    f.write(f"特征: {', '.join(selected_features)}\n\n")

    for name, (score, params, _) in sorted_results:
        f.write(f"\n{'='*70}\n")
        f.write(f"{name}\n")
        f.write(f"{'='*70}\n")
        f.write(f"准确率: {score:.4%}\n")
        f.write(f"参数:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")

print("\n调参结果已保存到: 批量调参结果_E1数据集.txt")
print("\n调参完成!")
