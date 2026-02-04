# -*- coding: utf-8 -*-
"""
综合可视化图表生成
包含：特征重要性条形图、多模型雷达图、ROC曲线、校准曲线、混淆矩阵
基于8个最优模型和E1数据集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import os
import warnings
warnings.filterwarnings('ignore')

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(script_dir)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 图表中使用英文标签（中文翻译）
# 训练集/验证集 -> Training Set / Validation Set
# 准确率 -> Accuracy
# 精确率 -> Precision
# 召回率 -> Recall
# F1-Score -> F1-Score
# 特异性 -> Specificity
# 假阳性率 -> False Positive Rate
# 真阳性率 -> True Positive Rate
# 随机分类器 -> Random Classifier
# 预测概率均值 -> Mean Predicted Probability
# 观测阳性率 -> Observed Positive Rate
# 完全校准 -> Perfect Calibration
# 预测类别 -> Predicted Class
# 真实类别 -> True Class
# 类别0 -> Class 0
# 类别1 -> Class 1

print("="*80)
print("综合可视化图表生成（8个最优模型）")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
data = pd.read_csv(os.path.join(project_dir, 'CARCT1_E1_FG2_processed_translated.csv'), encoding='utf-8-sig')

# 删除FCR_G2和ID为空的行（与SGD最优模型一致）
data = data.dropna(subset=['FCR_G2', 'ID'])

# 提取特征和标签
feature_cols = [col for col in data.columns if col not in ['ID', 'FCR_G2']]
X = data[feature_cols].copy()
y = data['FCR_G2'].copy()

# 删除全空列
X = X.dropna(axis=1, how='all')

# 删除缺失值超过50%的列（与LogisticRegression最优模型一致）
missing_ratio = X.isnull().sum() / len(X)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
X = X[cols_to_keep]

# 用中位数填充缺失值（与SGD最优模型一致）
X = X.fillna(X.median())

# 将类别标签从{1,2}转换为{0,1}
y_binary = (y - 1).astype(int)

print(f"样本数: {len(data)}")
print(f"特征数: {len(feature_cols)}")

# 2. 定义8个最优模型
print("\n【2. 定义8个最优模型】")

model_configs = {
    'SGD': {
        'model': SGDClassifier(loss='squared_hinge', penalty='l1', alpha=0.0038,
                            eta0=0.005, learning_rate='optimal', max_iter=5000,
                            random_state=42),
        'features': ['GAD7_0', 'Residence', 'Marriage', 'Education', 'Partner_Monthly_Income',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support', 'Chemotherapy',
                    'Perceived_Severity_of_Condition', 'Duration_Aware_of_Cancer_Diagnosis'],
        'scaler': RobustScaler(),
        'accuracy': '92.86%',
        'std': '±10.10%'
    },
    'LogisticRegression': {
        'model': LogisticRegression(C=1.0, penalty='l2', solver='saga',
                                 class_weight='balanced', max_iter=2000,
                                 tol=0.0001, random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '92.56%',
        'std': '±7.48%'
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '87.20%',
        'std': '±11.23%'
    },
    'SVM': {
        'model': SVC(C=0.1, kernel='linear', gamma='scale',
                   class_weight=None, probability=True, random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '88.99%',
        'std': '±9.59%'
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced', max_depth=None,
                                      max_features='sqrt', min_samples_leaf=1,
                                      min_samples_split=5, n_estimators=100, random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '85.12%',
        'std': '±10.15%'
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(max_depth=None, max_features='sqrt',
                                    min_samples_leaf=2, min_samples_split=10,
                                    n_estimators=100, random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '88.99%',
        'std': '±9.59%'
    },
    'NeuralNetwork': {
        'model': MLPClassifier(activation='logistic', alpha=0.0001,
                            hidden_layer_sizes=(50,), learning_rate='constant',
                            max_iter=1000, solver='adam', random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '88.99%',
        'std': '±13.93%'
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(learning_rate=0.01, max_depth=5,
                                          min_samples_leaf=4, min_samples_split=2,
                                          n_estimators=200, subsample=1.0, random_state=42),
        'features': ['GAD7_0', 'TCSQ_NC', 'Age', 'Residence', 'Education', 'Has_Partner',
                    'Relationship_with_Family', 'Family_Social_Emotional_Support',
                    'Perceived_Severity_of_Condition', 'Life_Economic_Stress'],
        'scaler': StandardScaler(),
        'accuracy': '84.82%',
        'std': '±11.04%'
    }
}

# 3. 准备模型数据
print("\n【3. 准备模型数据】")
model_data = {}
for model_name, config in model_configs.items():
    features = config['features']
    scaler = config['scaler']

    # 选择特征
    X_model = X[features].copy()

    # 标准化
    if scaler is not None:
        X_scaled = scaler.fit_transform(X_model)
    else:
        X_scaled = X_model.values

    model_data[model_name] = {
        'X': X_scaled,
        'features': features
    }

    print(f"{model_name}: {len(features)} features, scaler: {type(scaler).__name__}")

# 4. 计算交叉验证中的训练集和验证集性能指标
print("\n【4. 计算性能指标】")

# 使用8折分层交叉验证（与原最优参数一致）
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

def calculate_cv_train_val_metrics(model, X_scaled, y_true, cv):
    """
    在交叉验证中，收集每个fold的训练集和验证集的预测结果
    """
    y_train_all = []
    y_val_all = []
    y_train_pred_all = []
    y_val_pred_all = []
    y_train_prob_all = []
    y_val_prob_all = []

    for train_idx, val_idx in cv.split(X_scaled, y_true):
        X_train_fold = X_scaled[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_train_fold = y_true[train_idx]
        y_val_fold = y_true[val_idx]

        # 克隆模型并训练
        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)

        # 训练集预测
        y_train_pred = model_fold.predict(X_train_fold)
        if hasattr(model_fold, 'predict_proba'):
            y_train_prob = model_fold.predict_proba(X_train_fold)[:, 1]
        else:
            y_score = model_fold.decision_function(X_train_fold)
            if isinstance(y_score, list):
                y_score = np.array(y_score)
            y_train_prob = (y_score - float(np.min(y_score))) / (float(np.max(y_score)) - float(np.min(y_score)))

        # 验证集预测
        y_val_pred = model_fold.predict(X_val_fold)
        if hasattr(model_fold, 'predict_proba'):
            y_val_prob = model_fold.predict_proba(X_val_fold)[:, 1]
        else:
            y_score = model_fold.decision_function(X_val_fold)
            if isinstance(y_score, list):
                y_score = np.array(y_score)
            y_val_prob = (y_score - float(np.min(y_score))) / (float(np.max(y_score)) - float(np.min(y_score)))

        # 收集结果
        y_train_all.extend(y_train_fold)
        y_val_all.extend(y_val_fold)
        y_train_pred_all.extend(y_train_pred)
        y_val_pred_all.extend(y_val_pred)
        y_train_prob_all.extend(y_train_prob)
        y_val_prob_all.extend(y_val_prob)

    # 转换为numpy数组
    y_train_all = np.array(y_train_all)
    y_val_all = np.array(y_val_all)
    y_train_pred_all = np.array(y_train_pred_all)
    y_val_pred_all = np.array(y_val_pred_all)
    y_train_prob_all = np.array(y_train_prob_all)
    y_val_prob_all = np.array(y_val_prob_all)

    # 计算训练集指标
    cm_train = confusion_matrix(y_train_all, y_train_pred_all)
    tn, fp, fn, tp = cm_train.ravel()
    acc_train = (tp + tn) / (tp + tn + fp + fn)
    prec_train = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_train = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_train = 2 * prec_train * rec_train / (prec_train + rec_train) if (prec_train + rec_train) > 0 else 0
    spec_train = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算验证集指标
    cm_val = confusion_matrix(y_val_all, y_val_pred_all)
    tn, fp, fn, tp = cm_val.ravel()
    acc_val = (tp + tn) / (tp + tn + fp + fn)
    prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0
    spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0

    return (
        (acc_train, prec_train, rec_train, f1_train, spec_train, y_train_pred_all, y_train_prob_all, cm_train),
        (acc_val, prec_val, rec_val, f1_val, spec_val, y_val_pred_all, y_val_prob_all, cm_val)
    )

model_metrics_train = {}
model_metrics_val = {}

for model_name, config in model_configs.items():
    model = config['model']
    X_scaled = model_data[model_name]['X']

    # 计算交叉验证中的训练集和验证集性能
    train_result, val_result = calculate_cv_train_val_metrics(model, X_scaled, y_binary, skf)

    model_metrics_train[model_name] = {
        'accuracy': train_result[0], 'precision': train_result[1], 'recall': train_result[2],
        'f1': train_result[3], 'specificity': train_result[4],
        'y_pred': train_result[5], 'y_prob': train_result[6], 'cm': train_result[7]
    }

    model_metrics_val[model_name] = {
        'accuracy': val_result[0], 'precision': val_result[1], 'recall': val_result[2],
        'f1': val_result[3], 'specificity': val_result[4],
        'y_pred': val_result[5], 'y_prob': val_result[6], 'cm': val_result[7]
    }

    print(f"{model_name}: 训练集CV={train_result[0]*100:.2f}%, 验证集CV={val_result[0]*100:.2f}%")

# 按验证集准确率排序模型，LogisticRegression优先于SGD
def custom_sort_key(item):
    model_name, metrics = item
    # 定义优先级顺序，LR和SGD排在最前面
    priority = {'LogisticRegression': 0, 'SGD': 1}
    if model_name in priority:
        return (0, priority[model_name], -metrics['accuracy'])
    else:
        return (1, 0, -metrics['accuracy'])

sorted_models = sorted(model_metrics_val.items(), key=custom_sort_key)
model_names_sorted = [name for name, _ in sorted_models]
# 反转顺序，让排名靠前的模型后绘制（在上层）
model_names_plot = model_names_sorted[::-1]

# 创建颜色映射字典，确保每个模型始终使用相同的颜色
color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
model_colors = {model_name: color for model_name, color in zip(model_names_sorted, color_list)}
colors_for_plot = [model_colors[name] for name in model_names_plot]

# 5. 图表1：多模型性能比较雷达图（训练集）
print("\n【5. 生成多模型雷达图】")

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# 训练集雷达图
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

for i, (model_name, color) in enumerate(zip(model_names_plot, colors_for_plot)):
    metrics = model_metrics_train[model_name]
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
             metrics['f1'], metrics['specificity']]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f"{model_name}", color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
ax.grid(True, alpha=0.3)
ax.set_title('8 Model Performance Comparison (Training Set CV)', fontsize=14, fontweight='bold', pad=20)
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型性能比较（雷达图-训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型性能比较（雷达图-训练集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型性能比较（雷达图-训练集CV）.png")
print("已保存: 多模型性能比较（雷达图-训练集CV）.pdf")

# 验证集雷达图
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

for i, (model_name, color) in enumerate(zip(model_names_plot, colors_for_plot)):
    metrics = model_metrics_val[model_name]
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'],
             metrics['f1'], metrics['specificity']]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f"{model_name}", color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
ax.grid(True, alpha=0.3)
ax.set_title('8 Model Performance Comparison (Validation Set CV)', fontsize=14, fontweight='bold', pad=20)
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型性能比较（雷达图-验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型性能比较（雷达图-验证集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型性能比较（雷达图-验证集CV）.png")
print("已保存: 多模型性能比较（雷达图-验证集CV）.pdf")

# 6. 图表2：多模型ROC曲线对比（训练集）
print("\n【6. 生成多模型ROC曲线对比】")

fig, ax = plt.subplots(figsize=(10, 8))

roc_data = []

for model_name, color in zip(model_names_plot, colors_for_plot):
    model = model_configs[model_name]['model']

    # 训练集ROC（使用交叉验证中所有fold的训练集预测概率）
    y_prob = model_metrics_train[model_name]['y_prob']
    y_true = np.concatenate([y_binary[train_idx] for train_idx, _ in skf.split(model_data[model_name]['X'], y_binary)])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data.append((fpr, tpr, roc_auc, model_name, color))

    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.2f})", color=color)

ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('8 Model ROC Curve Comparison (Training Set CV)', fontsize=14, fontweight='bold')
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型ROC曲线对比（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型ROC曲线对比（训练集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型ROC曲线对比（训练集CV）.png")
print("已保存: 多模型ROC曲线对比（训练集CV）.pdf")

# 验证集ROC曲线
fig, ax = plt.subplots(figsize=(10, 8))

roc_data = []

for model_name, color in zip(model_names_plot, colors_for_plot):
    model = model_configs[model_name]['model']

    # 验证集ROC（使用交叉验证中所有fold的验证集预测概率）
    y_prob = model_metrics_val[model_name]['y_prob']
    y_true = np.concatenate([y_binary[val_idx] for _, val_idx in skf.split(model_data[model_name]['X'], y_binary)])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_data.append((fpr, tpr, roc_auc, model_name, color))

    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.2f})", color=color)

ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('8 Model ROC Curve Comparison (Validation Set CV)', fontsize=14, fontweight='bold')
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型ROC曲线对比（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型ROC曲线对比（验证集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型ROC曲线对比（验证集CV）.png")
print("已保存: 多模型ROC曲线对比（验证集CV）.pdf")

# 7. 图表3：多模型校准曲线对比（训练集）
print("\n【7. 生成多模型校准曲线对比】")

fig, ax = plt.subplots(figsize=(10, 8))

for model_name, color in zip(model_names_plot, colors_for_plot):
    y_prob = model_metrics_train[model_name]['y_prob']
    y_true = np.concatenate([y_binary[train_idx] for train_idx, _ in skf.split(model_data[model_name]['X'], y_binary)])

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            label=f"{model_name} (Brier={brier_score_loss(y_true, y_prob):.2f})",
            linewidth=2, color=color, markersize=4)

ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=1.5)
ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.1, color='gray')

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Observed Positive Rate", fontsize=12)
ax.set_title("8 Model Calibration Curve Comparison (Training Set CV)", fontsize=14, fontweight='bold')
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="best", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型校准曲线对比（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型校准曲线对比（训练集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型校准曲线对比（训练集CV）.png")
print("已保存: 多模型校准曲线对比（训练集CV）.pdf")

# 验证集校准曲线
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, color in zip(model_names_plot, colors_for_plot):
    y_prob = model_metrics_val[model_name]['y_prob']
    y_true = np.concatenate([y_binary[val_idx] for _, val_idx in skf.split(model_data[model_name]['X'], y_binary)])

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)

    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            label=f"{model_name} (Brier={brier_score_loss(y_true, y_prob):.2f})",
            linewidth=2, color=color, markersize=4)

ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", linewidth=1.5)
ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.1, color='gray')

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Observed Positive Rate", fontsize=12)
ax.set_title("8 Model Calibration Curve Comparison (Validation Set CV)", fontsize=14, fontweight='bold')
# 反转图例顺序，使排名靠前的模型显示在上方
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="best", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '多模型校准曲线对比（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '多模型校准曲线对比（验证集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 多模型校准曲线对比（验证集CV）.png")
print("已保存: 多模型校准曲线对比（验证集CV）.pdf")

# 8. 图表4：数据三线表（训练集和验证集）
print("\n【8. 生成数据三线表】")

# 计算交叉验证的95%置信区间
def calculate_cv_ci(cv_scores):
    """
    计算95%置信区间，使用t分布方法
    对fold均值计算置信区间，然后截断到[0%, 100%]范围
    返回格式：均值(下限, 上限)
    """
    from scipy import stats

    n = len(cv_scores)
    mean = np.mean(cv_scores)
    std = np.std(cv_scores, ddof=1)

    # 使用t分布计算95% CI的边界
    ci_margin = stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n)

    # 计算上下界（截断到[0,1]范围）
    lower = max(0, mean - ci_margin)
    upper = min(1, mean + ci_margin)

    # 返回格式：均值(下限, 上限)
    return f"{mean*100:.2f}({lower*100:.2f}, {upper*100:.2f})"

# 收集每个fold的AUC
def calculate_fold_auc(model_name, y_probs_list, y_true_list):
    """计算每个fold的AUC"""
    from sklearn.metrics import roc_auc_score
    auc_scores = []
    for y_prob, y_true in zip(y_probs_list, y_true_list):
        try:
            auc = roc_auc_score(y_true, y_prob)
            auc_scores.append(auc)
        except:
            pass
    return auc_scores

# 为每个模型收集训练集和验证集的fold准确率和AUC
from sklearn.metrics import roc_auc_score

# 重新运行CV获取每个fold的分数
cv_metrics = {}
for model_name in model_names_sorted:
    model = model_configs[model_name]['model']
    X_scaled = model_data[model_name]['X']
    
    train_acc_folds = []
    val_acc_folds = []
    train_auc_folds = []
    val_auc_folds = []
    
    for train_idx, val_idx in skf.split(X_scaled, y_binary):
        X_train_fold = X_scaled[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_train_fold = y_binary[train_idx]
        y_val_fold = y_binary[val_idx]
        
        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)
        
        # 训练集指标
        y_train_pred = model_fold.predict(X_train_fold)
        train_acc = (y_train_pred == y_train_fold).mean()
        train_acc_folds.append(train_acc)
        
        if hasattr(model_fold, 'predict_proba'):
            y_train_prob = model_fold.predict_proba(X_train_fold)[:, 1]
        else:
            y_score = model_fold.decision_function(X_train_fold)
            if isinstance(y_score, list):
                y_score = np.array(y_score)
            y_train_prob = (y_score - float(np.min(y_score))) / (float(np.max(y_score)) - float(np.min(y_score)))
        
        try:
            train_auc = roc_auc_score(y_train_fold, y_train_prob)
            train_auc_folds.append(train_auc)
        except:
            pass
        
        # 验证集指标
        y_val_pred = model_fold.predict(X_val_fold)
        val_acc = (y_val_pred == y_val_fold).mean()
        val_acc_folds.append(val_acc)
        
        if hasattr(model_fold, 'predict_proba'):
            y_val_prob = model_fold.predict_proba(X_val_fold)[:, 1]
        else:
            y_score = model_fold.decision_function(X_val_fold)
            if isinstance(y_score, list):
                y_score = np.array(y_score)
            y_val_prob = (y_score - float(np.min(y_score))) / (float(np.max(y_score)) - float(np.min(y_score)))
        
        try:
            val_auc = roc_auc_score(y_val_fold, y_val_prob)
            val_auc_folds.append(val_auc)
        except:
            pass
    
    cv_metrics[model_name] = {
        'train_acc': train_acc_folds,
        'val_acc': val_acc_folds,
        'train_auc': train_auc_folds,
        'val_auc': val_auc_folds
    }

table_data = []
for model_name in model_names_sorted:
    config = model_configs[model_name]
    metrics_train = model_metrics_train[model_name]
    metrics_val = model_metrics_val[model_name]
    scaler_name = type(config['scaler']).__name__
    
    # 计算95% CI
    train_acc_ci = calculate_cv_ci(cv_metrics[model_name]['train_acc'])
    val_acc_ci = calculate_cv_ci(cv_metrics[model_name]['val_acc'])
    train_auc_ci = calculate_cv_ci(cv_metrics[model_name]['train_auc'])
    val_auc_ci = calculate_cv_ci(cv_metrics[model_name]['val_auc'])
    
    table_data.append([
        model_name,
        len(config['features']),
        scaler_name,
        train_acc_ci,
        val_acc_ci,
        train_auc_ci,
        val_auc_ci,
        f"{metrics_train['precision']*100:.2f}%",
        f"{metrics_val['precision']*100:.2f}%",
        f"{metrics_train['recall']*100:.2f}%",
        f"{metrics_val['recall']*100:.2f}%",
        f"{metrics_train['f1']*100:.2f}%",
        f"{metrics_val['f1']*100:.2f}%"
    ])

df_table = pd.DataFrame(table_data, columns=['模型', '特征数',
                                          '标准化方法',
                                          '训练集准确率(95%CI)', '验证集准确率(95%CI)',
                                          '训练集AUC(95%CI)', '验证集AUC(95%CI)',
                                          '训练集精确率', '验证集精确率',
                                          '训练集召回率', '验证集召回率',
                                          '训练集F1', '验证集F1'])
df_table.to_csv(os.path.join(script_dir, '模型性能对比三线表.csv'), index=False, encoding='utf-8-sig')
print("已保存: 模型性能对比三线表.csv")

# 同时生成表格图片
fig, ax = plt.subplots(figsize=(22, 8))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_table.values.tolist(),
              colLabels=df_table.columns.tolist(),
              cellLoc='center',
              loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)

# 设置表头样式
for i in range(len(df_table.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置斑马纹
for i in range(1, len(df_table) + 1):
    for j in range(len(df_table.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

ax.set_title('8 Model Performance Comparison Table (Training Set vs Validation Set CV)', fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(script_dir, '模型性能对比三线表.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '模型性能对比三线表.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 模型性能对比三线表.png")
print("已保存: 模型性能对比三线表.pdf")

# 9. 图表5：所有模型混淆矩阵（2x4布局，训练集）
print("\n【9. 生成混淆矩阵组合图】")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (model_name, ax) in enumerate(zip(model_names_sorted, axes)):
    cm = model_metrics_train[model_name]['cm']
    acc = model_metrics_train[model_name]['accuracy']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)

    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'{model_name}\nTraining Set CV Accuracy={acc*100:.2f}%', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_yticklabels(['Class 0', 'Class 1'])

plt.suptitle('8 Model Confusion Matrix Comparison (Training Set CV)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '8模型混淆矩阵对比（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '8模型混淆矩阵对比（训练集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 8模型混淆矩阵对比（训练集CV）.png")
print("已保存: 8模型混淆矩阵对比（训练集CV）.pdf")

# 验证集混淆矩阵
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (model_name, ax) in enumerate(zip(model_names_sorted, axes)):
    cm = model_metrics_val[model_name]['cm']
    acc = model_metrics_val[model_name]['accuracy']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)

    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'{model_name}\nValidation Set CV Accuracy={acc*100:.2f}%', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_yticklabels(['Class 0', 'Class 1'])

plt.suptitle('8 Model Confusion Matrix Comparison (Validation Set CV)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '8模型混淆矩阵对比（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '8模型混淆矩阵对比（验证集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 8模型混淆矩阵对比（验证集CV）.png")
print("已保存: 8模型混淆矩阵对比（验证集CV）.pdf")

# 10. 图表6：性能指标条形图（训练集和测试集）
print("\n【10. 生成性能指标条形图】")

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity']

# 训练集条形图
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
    ax = axes[idx]

    values = [model_metrics_train[name][metric_key] * 100 for name in model_names_sorted]
    colors_for_bar = [model_colors[name] for name in model_names_sorted]

    bars = ax.bar(range(len(model_names_sorted)), values, color=colors_for_bar)
    ax.set_xticks(range(len(model_names_sorted)))
    ax.set_xticklabels(model_names_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Comparison (Training Set CV)', fontsize=12, fontweight='bold')
    ax.set_ylim([50, 100])
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

# 隐藏最后一个子图
axes[-1].axis('off')

plt.suptitle('8 Model Performance Metrics Comparison (Training Set CV)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '性能指标详细对比（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '性能指标详细对比（训练集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 性能指标详细对比（训练集CV）.png")
print("已保存: 性能指标详细对比（训练集CV）.pdf")

# 验证集条形图
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (metric_name, metric_key) in enumerate(zip(metrics_names, metrics_keys)):
    ax = axes[idx]

    values = [model_metrics_val[name][metric_key] * 100 for name in model_names_sorted]
    colors_for_bar = [model_colors[name] for name in model_names_sorted]

    bars = ax.bar(range(len(model_names_sorted)), values, color=colors_for_bar)
    ax.set_xticks(range(len(model_names_sorted)))
    ax.set_xticklabels(model_names_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Comparison (Validation Set CV)', fontsize=12, fontweight='bold')
    ax.set_ylim([50, 100])
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

# 隐藏最后一个子图
axes[-1].axis('off')

plt.suptitle('8 Model Performance Metrics Comparison (Validation Set CV)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '性能指标详细对比（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '性能指标详细对比（验证集CV）.pdf'), bbox_inches='tight')
plt.close()
print("已保存: 性能指标详细对比（验证集CV）.png")
print("已保存: 性能指标详细对比（验证集CV）.pdf")

# 11. 总结
print("\n" + "="*80)
print("可视化图表生成完成")
print("="*80)

generated_files = [
    "多模型性能比较（雷达图-训练集CV）.png",
    "多模型性能比较（雷达图-训练集CV）.pdf",
    "多模型性能比较（雷达图-验证集CV）.png",
    "多模型性能比较（雷达图-验证集CV）.pdf",
    "多模型ROC曲线对比（训练集CV）.png",
    "多模型ROC曲线对比（训练集CV）.pdf",
    "多模型ROC曲线对比（验证集CV）.png",
    "多模型ROC曲线对比（验证集CV）.pdf",
    "多模型校准曲线对比（训练集CV）.png",
    "多模型校准曲线对比（训练集CV）.pdf",
    "多模型校准曲线对比（验证集CV）.png",
    "多模型校准曲线对比（验证集CV）.pdf",
    "模型性能对比三线表.csv",
    "模型性能对比三线表.png",
    "模型性能对比三线表.pdf",
    "8模型混淆矩阵对比（训练集CV）.png",
    "8模型混淆矩阵对比（训练集CV）.pdf",
    "8模型混淆矩阵对比（验证集CV）.png",
    "8模型混淆矩阵对比（验证集CV）.pdf",
    "性能指标详细对比（训练集CV）.png",
    "性能指标详细对比（训练集CV）.pdf",
    "性能指标详细对比（验证集CV）.png",
    "性能指标详细对比（验证集CV）.pdf"
]

print("\n已生成的文件:")
for i, file in enumerate(generated_files, 1):
    print(f"  {i}. {file}")

print(f"\n总计: {len(generated_files)} 个文件")
print("\n" + "="*80)
print("完成")
print("="*80)
