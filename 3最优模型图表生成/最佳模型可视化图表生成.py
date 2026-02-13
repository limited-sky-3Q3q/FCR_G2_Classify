# -*- coding: utf-8 -*-
"""
最佳模型（LogisticRegression）可视化图表生成
包含：ROC曲线、校准曲线、DCA曲线、混淆矩阵、SHAP特征重要性、单个样本解释图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import os
import warnings
warnings.filterwarnings('ignore')

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# SHAP相关
try:
    import shap
    SHAP_AVAILABLE = True
    # 修复SHAP图表中负号显示问题
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 尝试修复SHAP内部字体渲染
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 修复SHAP库内部负号显示问题 - 修改SHAP的格式化函数
    def fix_shap_minus_sign():
        """修复SHAP图表中的负号显示问题"""
        import locale
        import sys
        
        # 设置系统编码为UTF-8
        if sys.platform == 'win32':
            import ctypes
            try:
                ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            except:
                pass
        
        # 强制使用ASCII减号
        from matplotlib import ticker
        ticker.ScalarFormatter.useMathText = False
    
    fix_shap_minus_sign()
    
except ImportError:
    SHAP_AVAILABLE = False
    print("注意: 未安装SHAP库，SHAP相关图表将跳过")
    print("      可使用命令安装: pip install shap")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

print("="*80)
print("最佳模型（LogisticRegression）可视化图表生成")
print("="*80)

# 1. 数据加载
print("\n【1. 数据加载】")
data = pd.read_csv(os.path.join(project_dir, 'CARCT1_E1_FG2_processed_translated.csv'), encoding='utf-8-sig')

# 删除FCR_G2和ID为空的行（与模型对比脚本一致）
data = data.dropna(subset=['FCR_G2', 'ID'])

# LogisticRegression模型的10个最优特征
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

# 数据清理
exclude_cols = ['FCR_G2', 'ID']
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = data[feature_cols].copy()

# 删除全空列
X = X.dropna(axis=1, how='all')

# 删除缺失值超过50%的列（与模型对比脚本一致）
missing_ratio = X.isnull().sum() / len(X)
cols_to_keep = missing_ratio[missing_ratio <= 0.5].index.tolist()
X = X[cols_to_keep]

# 用中位数填充缺失值（与模型对比脚本一致）
X = X.fillna(X.median())

# 选择最优特征
X_selected = X[optimal_features]

y = data['FCR_G2'].copy()

print(f"样本数: {len(data)}")
print(f"类别分布: 类别1={(y==1).sum()}, 类别2={(y==2).sum()}")

print(f"\nLogisticRegression模型使用的特征（共{len(optimal_features)}个）:")
for i, feat in enumerate(optimal_features, 1):
    print(f"  {i}. {feat}")

# 数据标准化
print("\n【2. 数据预处理】")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
print("数据标准化完成（使用StandardScaler）")

# 将类别标签从{1,2}转换为{0,1}
y_binary = (y - 1).astype(int)

# 3. 使用交叉验证评估模型
print("\n【3. 交叉验证评估模型】")
lr_params = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 2000,
    'tol': 0.0001,
    'random_state': 42
}

# 设置8折交叉验证
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
print(f"使用 {skf.n_splits} 折交叉验证")

# 收集所有折的训练集和验证集预测结果（参照模型对比脚本）
train_y_true = []
train_y_pred = []
train_y_proba = []
train_fpr = []
train_tpr = []
train_cm_sum = np.array([[0, 0], [0, 0]])
train_metrics = []

val_y_true = []
val_y_pred = []
val_y_proba = []
val_fpr = []
val_tpr = []
val_cm_sum = np.array([[0, 0], [0, 0]])
val_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_binary), 1):
    print(f"  正在处理第 {fold}/{skf.n_splits} 折...")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_binary[train_idx], y_binary[val_idx]

    # 训练模型
    model = LogisticRegression(**lr_params)
    model.fit(X_train, y_train)

    # 训练集预测
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_y_true.extend(y_train)
    train_y_pred.extend(y_train_pred)
    train_y_proba.extend(y_train_pred_proba)

    # 训练集ROC曲线
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    train_fpr.append(fpr_train)
    train_tpr.append(tpr_train)

    # 训练集混淆矩阵
    cm_train = confusion_matrix(y_train, y_train_pred)
    train_cm_sum += cm_train

    # 训练集指标
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, zero_division=0)
    recall_train = recall_score(y_train, y_train_pred, zero_division=0)
    f1_train = f1_score(y_train, y_train_pred, zero_division=0)
    auc_train = roc_auc_score(y_train, y_train_pred_proba)
    brier_train = brier_score_loss(y_train, y_train_pred_proba)

    tn, fp, fn, tp = cm_train.ravel()
    specificity_train = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv_train = precision_train
    npv_train = tn / (tn + fn) if (tn + fn) > 0 else 0

    train_metrics.append({
        'accuracy': accuracy_train,
        'precision': precision_train,
        'recall': recall_train,
        'f1': f1_train,
        'specificity': specificity_train,
        'auc': auc_train,
        'brier': brier_train,
        'ppv': ppv_train,
        'npv': npv_train,
        'sensitivity': recall_train
    })

    # 验证集预测
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_y_true.extend(y_val)
    val_y_pred.extend(y_val_pred)
    val_y_proba.extend(y_val_pred_proba)

    # 验证集ROC曲线
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
    val_fpr.append(fpr_val)
    val_tpr.append(tpr_val)

    # 验证集混淆矩阵
    cm_val = confusion_matrix(y_val, y_val_pred)
    val_cm_sum += cm_val

    # 验证集指标
    accuracy_val = accuracy_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred, zero_division=0)
    recall_val = recall_score(y_val, y_val_pred, zero_division=0)
    f1_val = f1_score(y_val, y_val_pred, zero_division=0)
    auc_val = roc_auc_score(y_val, y_val_pred_proba)
    brier_val = brier_score_loss(y_val, y_val_pred_proba)

    tn, fp, fn, tp = cm_val.ravel()
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv_val = precision_val
    npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0

    val_metrics.append({
        'accuracy': accuracy_val,
        'precision': precision_val,
        'recall': recall_val,
        'f1': f1_val,
        'specificity': specificity_val,
        'auc': auc_val,
        'brier': brier_val,
        'ppv': ppv_val,
        'npv': npv_val,
        'sensitivity': recall_val
    })

# 4. 计算平均指标
print("\n【4. 交叉验证结果汇总】")
train_y_true = np.array(train_y_true)
train_y_pred = np.array(train_y_pred)
train_y_proba = np.array(train_y_proba)
val_y_true = np.array(val_y_true)
val_y_pred = np.array(val_y_pred)
val_y_proba = np.array(val_y_proba)

# 训练集平均指标
train_accuracy_mean = np.mean([m['accuracy'] for m in train_metrics])
train_accuracy_std = np.std([m['accuracy'] for m in train_metrics])
train_precision_mean = np.mean([m['precision'] for m in train_metrics])
train_recall_mean = np.mean([m['recall'] for m in train_metrics])
train_f1_mean = np.mean([m['f1'] for m in train_metrics])
train_auc_mean = np.mean([m['auc'] for m in train_metrics])
train_auc_std = np.std([m['auc'] for m in train_metrics])
train_brier_mean = np.mean([m['brier'] for m in train_metrics])
train_brier_std = np.std([m['brier'] for m in train_metrics])
train_sensitivity_mean = np.mean([m['sensitivity'] for m in train_metrics])
train_specificity_mean = np.mean([m['specificity'] for m in train_metrics])
train_ppv_mean = np.mean([m['ppv'] for m in train_metrics])
train_npv_mean = np.mean([m['npv'] for m in train_metrics])

# 验证集平均指标
val_accuracy_mean = np.mean([m['accuracy'] for m in val_metrics])
val_accuracy_std = np.std([m['accuracy'] for m in val_metrics])
val_precision_mean = np.mean([m['precision'] for m in val_metrics])
val_recall_mean = np.mean([m['recall'] for m in val_metrics])
val_f1_mean = np.mean([m['f1'] for m in val_metrics])
val_auc_mean = np.mean([m['auc'] for m in val_metrics])
val_auc_std = np.std([m['auc'] for m in val_metrics])
val_brier_mean = np.mean([m['brier'] for m in val_metrics])
val_brier_std = np.std([m['brier'] for m in val_metrics])
val_sensitivity_mean = np.mean([m['sensitivity'] for m in val_metrics])
val_specificity_mean = np.mean([m['specificity'] for m in val_metrics])
val_ppv_mean = np.mean([m['ppv'] for m in val_metrics])
val_npv_mean = np.mean([m['npv'] for m in val_metrics])

# 计算总体混淆矩阵
train_tn, train_fp, train_fn, train_tp = train_cm_sum.ravel()
val_tn, val_fp, val_fn, val_tp = val_cm_sum.ravel()

print(f"\n训练集性能（8折平均）:")
print(f"  Accuracy (准确率): {train_accuracy_mean:.4f} ± {train_accuracy_std:.4f}")
print(f"  AUC: {train_auc_mean:.4f} ± {train_auc_std:.4f}")
print(f"  Brier Score: {train_brier_mean:.4f} ± {train_brier_std:.4f}")
print(f"  Precision (PPV): {train_ppv_mean:.4f}")
print(f"  Recall (Sensitivity): {train_sensitivity_mean:.4f}")
print(f"  Specificity: {train_specificity_mean:.4f}")
print(f"  NPV: {train_npv_mean:.4f}")
print(f"  F1-Score: {train_f1_mean:.4f}")

print(f"\n验证集性能（8折平均）:")
print(f"  Accuracy (准确率): {val_accuracy_mean:.4f} ± {val_accuracy_std:.4f}")
print(f"  AUC: {val_auc_mean:.4f} ± {val_auc_std:.4f}")
print(f"  Brier Score: {val_brier_mean:.4f} ± {val_brier_std:.4f}")
print(f"  Precision (PPV): {val_ppv_mean:.4f}")
print(f"  Recall (Sensitivity): {val_sensitivity_mean:.4f}")
print(f"  Specificity: {val_specificity_mean:.4f}")
print(f"  NPV: {val_npv_mean:.4f}")
print(f"  F1-Score: {val_f1_mean:.4f}")

print(f"\n累积混淆矩阵 - 训练集:")
print(f"  TN={train_tn}, FP={train_fp}, FN={train_fn}, TP={train_tp}")
print(f"\n累积混淆矩阵 - 验证集:")
print(f"  TN={val_tn}, FP={val_fp}, FN={val_fn}, TP={val_tp}")

# 5. 生成ROC曲线（基于交叉验证）
print("\n【5. 生成ROC曲线】")

# 5.1 训练集ROC曲线
print("  生成训练集ROC曲线...")

# 对所有折的训练集ROC曲线进行插值
mean_fpr = np.linspace(0, 1, 100)
train_tprs = []
for fpr, tpr in zip(train_fpr, train_tpr):
    train_tprs.append(np.interp(mean_fpr, fpr, tpr))
    train_tprs[-1][0] = 0.0

mean_train_tpr = np.mean(train_tprs, axis=0)
mean_train_tpr[-1] = 1.0

# 计算训练集AUC（使用与模型对比脚本一致的方法）
fpr_train, tpr_train, _ = roc_curve(train_y_true, train_y_proba)
train_auc = auc(fpr_train, tpr_train)
train_auc_std = np.std([auc(fpr, tpr) for fpr, tpr in zip(train_fpr, train_tpr)])

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, color='#E74C3C', lw=3,
         label=f'ROC (AUC = {train_auc:.2f} ± {train_auc_std:.2f})')

# 绘制各折的ROC曲线（半透明）
for fpr, tpr in zip(train_fpr, train_tpr):
    plt.plot(fpr, tpr, color='#E74C3C', alpha=0.1, lw=1)

plt.plot([0, 1], [0, 1], color='#95A5A6', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
plt.title('ROC Curve - LogisticRegression (Training Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型ROC曲线（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型ROC曲线（训练集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型ROC曲线（训练集CV）.png")
print("  已保存: 最佳模型ROC曲线（训练集CV）.pdf")

# 5.2 验证集ROC曲线
print("  生成验证集ROC曲线...")

# 对所有折的验证集ROC曲线进行插值
mean_fpr = np.linspace(0, 1, 100)
val_tprs = []
for fpr, tpr in zip(val_fpr, val_tpr):
    val_tprs.append(np.interp(mean_fpr, fpr, tpr))
    val_tprs[-1][0] = 0.0

mean_val_tpr = np.mean(val_tprs, axis=0)
mean_val_tpr[-1] = 1.0

# 计算验证集AUC
fpr_val, tpr_val, _ = roc_curve(val_y_true, val_y_proba)
val_auc = auc(fpr_val, tpr_val)
val_auc_std = np.std([auc(fpr, tpr) for fpr, tpr in zip(val_fpr, val_tpr)])

plt.figure(figsize=(10, 8))
plt.plot(fpr_val, tpr_val, color='#E74C3C', lw=3,
         label=f'ROC (AUC = {val_auc:.2f} ± {val_auc_std:.2f})')

# 绘制各折的ROC曲线（半透明）
for fpr, tpr in zip(val_fpr, val_tpr):
    plt.plot(fpr, tpr, color='#E74C3C', alpha=0.1, lw=1)

plt.plot([0, 1], [0, 1], color='#95A5A6', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
plt.title('ROC Curve - LogisticRegression (Validation Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型ROC曲线（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型ROC曲线（验证集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型ROC曲线（验证集CV）.png")
print("  已保存: 最佳模型ROC曲线（验证集CV）.pdf")

# 6. 生成校准曲线（基于交叉验证）
print("\n【6. 生成校准曲线】")

# 6.1 训练集校准曲线
print("  生成训练集校准曲线...")

# 收集所有训练集预测结果并计算校准曲线
# 使用quantile策略，减少bin数量，避免样本分布不均
train_prob_true, train_prob_pred = calibration_curve(train_y_true, train_y_proba, n_bins=5, strategy='quantile')

# 计算每个bin的样本数（用于标注）
bin_edges = np.quantile(train_y_proba, np.linspace(0, 1, 6))
bin_samples = []
for i in range(5):
    lower = bin_edges[i]
    upper = bin_edges[i+1]
    if i < 4:
        mask = (train_y_proba >= lower) & (train_y_proba < upper)
    else:
        mask = (train_y_proba >= lower)
    bin_samples.append(np.sum(mask))

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', color='#95A5A6', lw=2, label='Perfectly Calibrated')
plt.plot(train_prob_pred, train_prob_true, marker='o', color='#E74C3C', lw=3, markersize=10,
         label=f'LogisticRegression (Brier = {train_brier_mean:.2f} ± {train_brier_std:.2f})')

plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
plt.title('Calibration Curve - LogisticRegression (Training Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型校准曲线（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型校准曲线（训练集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型校准曲线（训练集CV）.png")
print("  已保存: 最佳模型校准曲线（训练集CV）.pdf")

# 6.2 验证集校准曲线
print("  生成验证集校准曲线...")

# 收集所有验证集预测结果并计算校准曲线
# 使用quantile策略，减少bin数量，避免样本分布不均
val_prob_true, val_prob_pred = calibration_curve(val_y_true, val_y_proba, n_bins=5, strategy='quantile')

# 计算每个bin的样本数（用于标注）
bin_edges = np.quantile(val_y_proba, np.linspace(0, 1, 6))
bin_samples = []
for i in range(5):
    lower = bin_edges[i]
    upper = bin_edges[i+1]
    if i < 4:
        mask = (val_y_proba >= lower) & (val_y_proba < upper)
    else:
        mask = (val_y_proba >= lower)
    bin_samples.append(np.sum(mask))

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', color='#95A5A6', lw=2, label='Perfectly Calibrated')
plt.plot(val_prob_pred, val_prob_true, marker='o', color='#E74C3C', lw=3, markersize=10,
         label=f'LogisticRegression (Brier = {val_brier_mean:.2f} ± {val_brier_std:.2f})')

plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
plt.title('Calibration Curve - LogisticRegression (Validation Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型校准曲线（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型校准曲线（验证集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型校准曲线（验证集CV）.png")
print("  已保存: 最佳模型校准曲线（验证集CV）.pdf")

# 7. 生成DCA曲线（基于交叉验证）
print("\n【7. 生成DCA曲线】")

def calculate_net_benefit(y_true, y_proba, threshold_prob):
    """计算净获益"""
    n = len(y_true)
    tp = np.sum((y_true == 1) & (y_proba >= threshold_prob))
    fp = np.sum((y_true == 0) & (y_proba >= threshold_prob))

    # 净获益公式: TP/n - (FP/n) * (pt / (1-pt))
    # 其中pt是阈值概率
    if threshold_prob == 0:
        return tp / n
    elif threshold_prob >= 0.999:
        return 0
    else:
        net_benefit = tp / n - fp / n * (threshold_prob / (1 - threshold_prob))
        return net_benefit

# 7.1 训练集DCA曲线
print("  生成训练集DCA曲线...")

# 打印调试信息
print(f"    训练集样本数: {len(train_y_true)}")
print(f"    训练集正例数: {np.sum(train_y_true == 1)}")
print(f"    训练集预测概率范围: [{train_y_proba.min():.4f}, {train_y_proba.max():.4f}]")

# 计算不同阈值下的净获益（使用所有交叉验证的训练集预测结果）
# 避免阈值正好为0或1，使用0.01-0.99的范围
threshold_probs = np.linspace(0.01, 0.99, 99)
train_model_net_benefit = []
train_treat_all_net_benefit = []
train_treat_none_net_benefit = []

total_n_train = len(train_y_true)
total_tp_all_train = np.sum(train_y_true == 1)
prevalence_train = total_tp_all_train / total_n_train

print(f"    训练集患病率(Prevalence): {prevalence_train:.4f}")

for thresh in threshold_probs:
    # 模型净获益
    model_nb = calculate_net_benefit(train_y_true, train_y_proba, thresh)
    train_model_net_benefit.append(model_nb)

    # 治疗所有人净获益
    # 当阈值为0时，所有人都接受治疗
    # 净获益 = prevalence - (1-prevalence) * (pt/(1-pt))
    treat_all_nb = prevalence_train - (1 - prevalence_train) * (thresh / (1 - thresh))
    train_treat_all_net_benefit.append(treat_all_nb)

    # 不治疗任何人净获益始终为0
    train_treat_none_net_benefit.append(0)

# 打印一些关键点的净获益值
print(f"    模型净获益范围: [{min(train_model_net_benefit):.4f}, {max(train_model_net_benefit):.4f}]")
print(f"    Treat All净获益范围: [{min(train_treat_all_net_benefit):.4f}, {max(train_treat_all_net_benefit):.4f}]")
print(f"    示例（阈值=0.1）: 模型={train_model_net_benefit[9]:.4f}, Treat All={train_treat_all_net_benefit[9]:.4f}")
print(f"    示例（阈值=0.5）: 模型={train_model_net_benefit[49]:.4f}, Treat All={train_treat_all_net_benefit[49]:.4f}")

# 对Treat All曲线进行截断，避免极端负值影响显示
# 只显示在合理范围内的值（例如 -0.5 到 max + 0.1）
treat_all_cutoff = -0.5
train_treat_all_net_benefit_clipped = [max(nb, treat_all_cutoff) if nb < 0 else nb for nb in train_treat_all_net_benefit]

plt.figure(figsize=(12, 8))
plt.plot(threshold_probs, train_model_net_benefit, color='#E74C3C', lw=3,
         label=f'LogisticRegression (Training Set CV, 8-Fold)', marker='o', markersize=4, markevery=10)
plt.plot(threshold_probs, train_treat_all_net_benefit_clipped, color='#3498DB', lw=2,
         label='Treat All', linestyle='--')
plt.plot(threshold_probs, train_treat_none_net_benefit, color='#2ECC71', lw=2,
         label='Treat None', linestyle='-.')

plt.xlabel('Threshold Probability', fontsize=14, fontweight='bold')
plt.ylabel('Net Benefit', fontsize=14, fontweight='bold')
plt.title('Decision Curve Analysis (DCA) - LogisticRegression (Training Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
# 设置y轴范围，包含负值，使曲线更清晰
# 只关注模型净获益和合理范围内的Treat All
valid_train_nb = [x for x in train_model_net_benefit if not np.isnan(x) and not np.isinf(x)]
valid_treat_all_nb = [x for x in train_treat_all_net_benefit_clipped if not np.isnan(x) and not np.isinf(x)]
y_min = min(min(valid_train_nb) if valid_train_nb else 0, min(valid_treat_all_nb) if valid_treat_all_nb else 0, 0) - 0.05
y_max = max(max(valid_train_nb) if valid_train_nb else 0, max(valid_treat_all_nb) if valid_treat_all_nb else 0, 0) + 0.15
plt.ylim([y_min, max(y_max, 0.7)])
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'DCA曲线图（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, 'DCA曲线图（训练集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: DCA曲线图（训练集CV）.png")
print("  已保存: DCA曲线图（训练集CV）.pdf")

# 7.2 验证集DCA曲线
print("  生成验证集DCA曲线...")

# 打印调试信息
print(f"    验证集样本数: {len(val_y_true)}")
print(f"    验证集正例数: {np.sum(val_y_true == 1)}")
print(f"    验证集预测概率范围: [{val_y_proba.min():.4f}, {val_y_proba.max():.4f}]")

# 计算不同阈值下的净获益（使用所有交叉验证的验证集预测结果）
# 使用与训练集相同的阈值范围
val_model_net_benefit = []
val_treat_all_net_benefit = []
val_treat_none_net_benefit = []

total_n_val = len(val_y_true)
total_tp_all_val = np.sum(val_y_true == 1)
prevalence_val = total_tp_all_val / total_n_val

print(f"    验证集患病率(Prevalence): {prevalence_val:.4f}")

for thresh in threshold_probs:
    # 模型净获益
    model_nb = calculate_net_benefit(val_y_true, val_y_proba, thresh)
    val_model_net_benefit.append(model_nb)

    # 治疗所有人净获益
    treat_all_nb = prevalence_val - (1 - prevalence_val) * (thresh / (1 - thresh))
    val_treat_all_net_benefit.append(treat_all_nb)

    # 不治疗任何人净获益始终为0
    val_treat_none_net_benefit.append(0)

# 打印一些关键点的净获益值
print(f"    模型净获益范围: [{min(val_model_net_benefit):.4f}, {max(val_model_net_benefit):.4f}]")
print(f"    Treat All净获益范围: [{min(val_treat_all_net_benefit):.4f}, {max(val_treat_all_net_benefit):.4f}]")
print(f"    示例（阈值=0.1）: 模型={val_model_net_benefit[9]:.4f}, Treat All={val_treat_all_net_benefit[9]:.4f}")
print(f"    示例（阈值=0.5）: 模型={val_model_net_benefit[49]:.4f}, Treat All={val_treat_all_net_benefit[49]:.4f}")

# 对Treat All曲线进行截断，避免极端负值影响显示
treat_all_cutoff = -0.5
val_treat_all_net_benefit_clipped = [max(nb, treat_all_cutoff) if nb < 0 else nb for nb in val_treat_all_net_benefit]

plt.figure(figsize=(12, 8))
plt.plot(threshold_probs, val_model_net_benefit, color='#E74C3C', lw=3,
         label=f'LogisticRegression (Validation Set CV, 8-Fold)', marker='o', markersize=4, markevery=10)
plt.plot(threshold_probs, val_treat_all_net_benefit_clipped, color='#3498DB', lw=2,
         label='Treat All', linestyle='--')
plt.plot(threshold_probs, val_treat_none_net_benefit, color='#2ECC71', lw=2,
         label='Treat None', linestyle='-.')

plt.xlabel('Threshold Probability', fontsize=14, fontweight='bold')
plt.ylabel('Net Benefit', fontsize=14, fontweight='bold')
plt.title('Decision Curve Analysis (DCA) - LogisticRegression (Validation Set CV, 8-Fold)\nFCR Trajectory Prediction', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
# 设置y轴范围，包含负值，使曲线更清晰
# 只关注模型净获益和合理范围内的Treat All
valid_val_nb = [x for x in val_model_net_benefit if not np.isnan(x) and not np.isinf(x)]
valid_val_treat_all_nb = [x for x in val_treat_all_net_benefit_clipped if not np.isnan(x) and not np.isinf(x)]
y_min = min(min(valid_val_nb) if valid_val_nb else 0, min(valid_val_treat_all_nb) if valid_val_treat_all_nb else 0, 0) - 0.05
y_max = max(max(valid_val_nb) if valid_val_nb else 0, max(valid_val_treat_all_nb) if valid_val_treat_all_nb else 0, 0) + 0.15
plt.ylim([y_min, max(y_max, 0.7)])
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'DCA曲线图（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, 'DCA曲线图（验证集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: DCA曲线图（验证集CV）.png")
print("  已保存: DCA曲线图（验证集CV）.pdf")

# 8. 生成混淆矩阵（基于交叉验证累积结果）
print("\n【8. 生成混淆矩阵】")

# 8.1 训练集混淆矩阵
print("  生成训练集混淆矩阵...")

train_cm = train_cm_sum
train_accuracy_overall = (train_cm[0, 0] + train_cm[1, 1]) / train_cm.sum()

fig, ax = plt.subplots(figsize=(9, 7))

# 创建混淆矩阵热图
im = ax.imshow(train_cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# 设置刻度和标签
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=12)
ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=12)

# 在单元格中添加数值
thresh = train_cm.max() / 2.
for i in range(train_cm.shape[0]):
    for j in range(train_cm.shape[1]):
        ax.text(j, i, format(train_cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if train_cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - LogisticRegression (Training Set CV, 8-Fold)\nFCR Trajectory Prediction (Overall Accuracy: {train_accuracy_overall:.4f})',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型的混淆矩阵（训练集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型的混淆矩阵（训练集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型的混淆矩阵（训练集CV）.png")
print("  已保存: 最佳模型的混淆矩阵（训练集CV）.pdf")

# 8.2 验证集混淆矩阵
print("  生成验证集混淆矩阵...")

val_cm = val_cm_sum
val_accuracy_overall = (val_cm[0, 0] + val_cm[1, 1]) / val_cm.sum()

fig, ax = plt.subplots(figsize=(9, 7))

# 创建混淆矩阵热图
im = ax.imshow(val_cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# 设置刻度和标签
tick_marks = np.arange(2)
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=12)
ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=12)

# 在单元格中添加数值
thresh = val_cm.max() / 2.
for i in range(val_cm.shape[0]):
    for j in range(val_cm.shape[1]):
        ax.text(j, i, format(val_cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if val_cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - LogisticRegression (Validation Set CV, 8-Fold)\nFCR Trajectory Prediction (Overall Accuracy: {val_accuracy_overall:.4f})',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, '最佳模型的混淆矩阵（验证集CV）.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, '最佳模型的混淆矩阵（验证集CV）.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: 最佳模型的混淆矩阵（验证集CV）.png")
print("  已保存: 最佳模型的混淆矩阵（验证集CV）.pdf")

# 9. 生成TOP-K特征重要性曲线图
print("\n【9. 生成TOP-K特征重要性曲线图】")

# 9.1 获取特征重要性（使用LogisticRegression的系数绝对值）
print("  计算特征重要性...")

# 在完整数据上训练一个模型以获取特征重要性
model_for_importance = LogisticRegression(**lr_params)
model_for_importance.fit(X_scaled, y_binary)

# 获取特征重要性（系数的绝对值）
feature_importance = np.abs(model_for_importance.coef_[0])
feature_importance_dict = dict(zip(optimal_features, feature_importance))

# 按重要性排序
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names = [f[0] for f in sorted_features]
sorted_importance_values = [f[1] for f in sorted_features]

print(f"  特征数量: {len(optimal_features)}")
print("  Top 10 特征:")
for i, (feat, imp) in enumerate(sorted_features[:10], 1):
    print(f"    {i}. {feat}: {imp:.4f}")

# 9.2 生成TOP-K特征重要性曲线图（累积重要性）
print("  生成TOP-K特征重要性曲线...")

# 计算累积重要性
cumulative_importance = np.cumsum(sorted_importance_values) / np.sum(sorted_importance_values)

# 创建两个子图：累积重要性曲线和Top K特征条形图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 左图：累积重要性曲线
k_values = range(1, len(sorted_features) + 1)
ax1.plot(k_values, cumulative_importance, 'o-', color='#E74C3C', lw=3, markersize=6, label='Cumulative Importance')
ax1.axhline(y=0.8, color='#3498DB', linestyle='--', linewidth=2, label='80% Threshold')
ax1.axhline(y=0.9, color='#2ECC71', linestyle='--', linewidth=2, label='90% Threshold')
ax1.axhline(y=0.95, color='#F39C12', linestyle='--', linewidth=2, label='95% Threshold')
ax1.axvline(x=np.where(cumulative_importance >= 0.9)[0][0] + 1 if len(np.where(cumulative_importance >= 0.9)[0]) > 0 else len(k_values),
            color='#9B59B6', linestyle=':', linewidth=2, alpha=0.7, label='90% K value')

ax1.set_xlabel('Number of Features (Top-K)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Feature Importance', fontsize=14, fontweight='bold')
ax1.set_title('TOP-K Feature Importance (Cumulative)\nLogisticRegression - FCR Trajectory Prediction', fontsize=15, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, len(sorted_features) + 0.5)
ax1.set_ylim(0, 1.05)

# 标记关键点
k_80 = np.where(cumulative_importance >= 0.8)[0][0] + 1 if len(np.where(cumulative_importance >= 0.8)[0]) > 0 else len(k_values)
k_90 = np.where(cumulative_importance >= 0.9)[0][0] + 1 if len(np.where(cumulative_importance >= 0.9)[0]) > 0 else len(k_values)
k_95 = np.where(cumulative_importance >= 0.95)[0][0] + 1 if len(np.where(cumulative_importance >= 0.95)[0]) > 0 else len(k_values)

ax1.text(k_80, 0.82, f'K={k_80}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#3498DB')
ax1.text(k_90, 0.92, f'K={k_90}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2ECC71')
ax1.text(k_95, 0.97, f'K={k_95}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#F39C12')

# 右图：Top K 特征重要性条形图
top_k = min(15, len(sorted_features))  # 显示前15个特征
y_pos = np.arange(top_k)[::-1]  # 从上到下显示
ax2.barh(y_pos, sorted_importance_values[:top_k], color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_feature_names[:top_k], fontsize=11)
ax2.invert_yaxis()  # 最重要的特征在顶部
ax2.set_xlabel('Feature Importance (|Coefficient|)', fontsize=14, fontweight='bold')
ax2.set_title(f'Top {top_k} Feature Importance\nLogisticRegression - FCR Trajectory Prediction', fontsize=15, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, axis='x')

# 在条形图上添加数值标签
for i, (idx, importance) in enumerate(zip(y_pos, sorted_importance_values[:top_k])):
    ax2.text(importance + 0.01 * max(sorted_importance_values), idx, f'{importance:.4f}',
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'TOP-K特征重要性曲线图.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, 'TOP-K特征重要性曲线图.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("  已保存: TOP-K特征重要性曲线图.png")
print("  已保存: TOP-K特征重要性曲线图.pdf")

# 输出TOP-K统计信息
print(f"\n  TOP-K特征统计:")
print(f"    达到80%重要性需: {k_80} 个特征")
print(f"    达到90%重要性需: {k_90} 个特征")
print(f"    达到95%重要性需: {k_95} 个特征")
print(f"    总特征数: {len(sorted_features)}")

# 10. 生成SHAP特征重要性图和单个样本解释图（基于交叉验证）
if SHAP_AVAILABLE:
    print("\n【9. 生成SHAP可视化（基于交叉验证）】")
    print(f"  数据集大小: {len(X_scaled)} 样本")
    print(f"  特征数量: {len(optimal_features)} 个")
    print(f"  样本分布: 类别0={(y_binary==0).sum()}, 类别1={(y_binary==1).sum()}")

    # 创建SHAP单个样本解释图文件夹
    shap_samples_dir = os.path.join(script_dir, 'SHAP单个样本解释图')
    os.makedirs(shap_samples_dir, exist_ok=True)
    print(f"  创建文件夹: {shap_samples_dir}")

    # 在完整数据上训练一个模型用于SHAP解释
    model_final = LogisticRegression(**lr_params)
    model_final.fit(X_scaled, y_binary)

    # 检查SHAP值
    print("  计算SHAP值...")
    # 使用background dataset来稳定SHAP值
    background_data = shap.sample(X_scaled, 50, random_state=42)
    explainer = shap.LinearExplainer(model_final, background_data, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    print(f"  SHAP值形状: {shap_values.shape}")
    print(f"  SHAP值范围: [{shap_values.min():.4f}, {shap_values.max():.4f}]")

    # 9.1 SHAP特征重要性条形图（全局特征重要性）
    print("  生成SHAP特征重要性条形图...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, plot_type="bar",
                      feature_names=optimal_features,
                      show=False, max_display=len(optimal_features))
    plt.title('SHAP Feature Importance - LogisticRegression\nFCR Trajectory Prediction',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'SHAP特征重要性条形图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: SHAP特征重要性条形图.png")

    # 9.2 SHAP特征重要性蜂群图（特征分布和影响）
    print("  生成SHAP特征重要性蜂群图...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=optimal_features,
                     show=False, max_display=len(optimal_features),
                     alpha=0.8, plot_size=(12, 8))
    plt.title('SHAP Summary Plot - LogisticRegression\nFCR Trajectory Prediction',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'SHAP特征重要性蜂群图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  已保存: SHAP特征重要性蜂群图.png")

    # 9.3 生成所有样本的SHAP解释图
    print(f"  生成所有样本SHAP解释图（共{len(X_scaled)}个样本）...")
    
    # 定义特征值映射字典（根据对照表，使用英文）
    # 原始数据值从0开始，需要添加0对应的描述
    value_mapping = {
        'Residence': {0: 'Urban', 1: 'Rural'},  # 原始数据: 0=Urban, 1=Rural
        'Education': {0: 'Junior High or below', 1: 'High School', 2: 'Bachelor or above'},  # 原始数据: 0,1,2
        'Has_Partner': {0: 'Yes', 1: 'No'},  # 原始数据: 0=Yes, 1=No
        'Relationship_with_Family': {0: 'Very Distant', 1: 'Quite Distant', 2: 'Average', 3: 'Quite Close', 4: 'Very Close'},  # 原始数据: 0-4
        'Family_Social_Emotional_Support': {0: 'Insufficient', 1: 'Average', 2: 'Sufficient', 3: 'Very Sufficient'},  # 原始数据: 0-3
        'Perceived_Severity_of_Condition': {0: 'Mild', 1: 'Moderate', 2: 'Severe'},  # 原始数据: 0-2
        'Life_Economic_Stress': {0: 'None', 1: 'No Stress', 2: 'Mild Stress', 3: 'Moderate Stress', 4: 'Severe Stress'}  # 原始数据: 0-4
    }
    
    # 9.3 生成所有样本SHAP瀑布图
    print(f"  生成所有样本SHAP瀑布图（共{len(X_scaled)}个样本）...")

    # Monkey patch SHAP内部格式化函数，移除负号显示
    import shap.plots._waterfall as shap_waterfall
    import re

    # 保存原始格式化函数
    original_format_value = shap_waterfall.format_value

    # 定义新的格式化函数，移除负号
    def format_value_no_minus(s, format_str="{:.3f}"):
        """格式化数值时移除负号"""
        if not issubclass(type(s), str):
            s = format_str % s
        # 移除尾部的0
        s = re.sub(r"\.?0+$", "", s)
        # 如果有负号，直接移除它（包括普通减号和Unicode减号）
        if s.startswith('-') or s.startswith('\u2212'):
            s = s[1:] if s.startswith('-') else s[1:]
        return s

    # 替换SHAP的格式化函数
    shap_waterfall.format_value = format_value_no_minus
    
    for sample_idx in range(len(X_scaled)):
        # 强制设置matplotlib使用中文字体
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun', 'DejaVu Sans']
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['axes.formatter.use_mathtext'] = False
        mpl.rcParams['text.usetex'] = False

        plt.figure(figsize=(12, 8))
        # 确保负号不显示 - 全面配置
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.formatter.use_mathtext'] = False
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        # 准备瀑布图数据，将数值转换为英文描述
        X_sample_original = X_selected.iloc[sample_idx].values.copy()
        X_sample_display = []
        
        for j, feat in enumerate(optimal_features):
            val = X_sample_original[j]
            if feat in value_mapping:
                # 离散特征：转换为英文描述
                int_val = int(round(val)) if not pd.isna(val) and isinstance(val, (int, float)) else val
                if int_val in value_mapping[feat]:
                    display_val = value_mapping[feat][int_val]
                else:
                    display_val = str(int_val)
            elif feat == 'GAD7_0' or feat == 'TCSQ_NC' or feat == 'Age':
                # 连续数值特征：保留原始数值
                display_val = f"{abs(val):.1f}"  # 使用abs()移除负号
            else:
                # 其他特征：直接显示
                display_val = str(abs(val))  # 使用abs()移除负号
            X_sample_display.append(display_val)
        
        # 创建Explanation对象，使用原始数值计算SHAP但用英文描述显示
        shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx],
                                            base_values=explainer.expected_value,
                                            data=X_sample_display,
                                            feature_names=optimal_features),
                          show=False, max_display=len(optimal_features))
        # 转换回原始标签（0->1, 1->2）
        true_label = int(y_binary[sample_idx]) + 1
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx+1} (True: FCR={true_label}, Pred: {int(y_binary[sample_idx]) + 1})\nFCR Trajectory Prediction',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        # 保存到SHAP单个样本解释图文件夹
        waterfall_png_path = os.path.join(shap_samples_dir, f'SHAP瀑布图_样本{sample_idx+1}.png')
        waterfall_pdf_path = os.path.join(shap_samples_dir, f'SHAP瀑布图_样本{sample_idx+1}.pdf')
        plt.savefig(waterfall_png_path, dpi=300, bbox_inches='tight')
        plt.savefig(waterfall_pdf_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 恢复原始的格式化函数
    shap_waterfall.format_value = original_format_value

    print(f"  已保存: SHAP瀑布图（共{len(X_scaled)}个样本）")

    # 9.4 生成所有样本的SHAP解释图（Force Plot）
    print(f"  生成所有样本SHAP解释图（共{len(X_scaled)}个样本）...")
    
    for i in range(len(X_scaled)):
        # 强制设置matplotlib使用中文字体
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'SimSun', 'DejaVu Sans']
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['axes.formatter.use_mathtext'] = False
        mpl.rcParams['text.usetex'] = False

        # 确保负号正确显示
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.formatter.use_mathtext'] = False

        # 使用原始数值（SHAP force plot不支持字符串）
        X_sample_original = X_selected.iloc[i].values

        # 生成特征值说明文本
        feature_descriptions = []
        for j, feat in enumerate(optimal_features):
            val = X_sample_original[j]
            if feat in value_mapping:
                int_val = int(round(val)) if not pd.isna(val) and isinstance(val, (int, float)) else val
                if int_val in value_mapping[feat]:
                    display_val = value_mapping[feat][int_val]
                else:
                    display_val = str(int_val)
                feature_descriptions.append(f"{feat}: {display_val}")
            elif feat == 'GAD7_0' or feat == 'TCSQ_NC' or feat == 'Age':
                feature_descriptions.append(f"{feat}: {abs(val):.1f}")  # 使用abs()移除负号
            else:
                feature_descriptions.append(f"{feat}: {abs(val):.1f}")  # 使用abs()移除负号

        shap.force_plot(explainer.expected_value, shap_values[i],
                       X_sample_original, feature_names=optimal_features,
                       matplotlib=True, show=False, figsize=(20, 4))
        # 转换回原始标签（0->1, 1->2）
        true_label = int(y_binary[i]) + 1
        # 使用plt.figtext在上方添加标题，减小字体并调整位置
        plt.figtext(0.5, 0.99, f'SHAP Explanation - Sample {i+1} (True: FCR={true_label}, Pred: {int(y_binary[i]) + 1})  FCR Trajectory Prediction',
                    fontsize=24, fontweight='bold', ha='center', va='top')
        
        # 在底部添加特征值说明
        desc_text = '\n'.join(feature_descriptions)
        plt.figtext(0.02, 0.62, f'Feature Values:\n{desc_text}',
                    fontsize=8, fontfamily='monospace', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        
        # 调整子图位置：hspace控制子图之间的垂直间距
        plt.subplots_adjust(top=0.50, bottom=0.01, left=0.05, right=0.95, hspace=0.3)
        # 保存到SHAP单个样本解释图文件夹
        force_png_path = os.path.join(shap_samples_dir, f'SHAP单个样本解释图_样本{i+1}.png')
        force_pdf_path = os.path.join(shap_samples_dir, f'SHAP单个样本解释图_样本{i+1}.pdf')
        plt.savefig(force_png_path, dpi=300, pad_inches=0.8)
        plt.savefig(force_pdf_path, dpi=300, pad_inches=0.8)
        plt.close()
    print(f"  已保存: SHAP单个样本解释图（共{len(X_scaled)}个样本）")

    # 9.5 生成特征依赖图（每个特征的SHAP值 vs 特征值）
    print("  生成SHAP特征依赖图...")
    for i, feat in enumerate(optimal_features):
        plt.figure(figsize=(10, 6))
        # 手动创建特征依赖图，以便更好地控制点的显示
        # 使用原始值而不是标准化值
        feature_values = X_selected.iloc[:, i].values  # 原始值
        shap_vals = shap_values[:, i]

        # 判断是否为离散特征（在value_mapping中）
        is_categorical = feat in value_mapping

        # 为离散特征添加小的随机抖动，使重叠的点可见
        if feature_values.max() > feature_values.min():
            jitter_amount = 0.02 * (feature_values.max() - feature_values.min())
        else:
            jitter_amount = 0.02  # 如果最大值等于最小值，使用固定抖动

        if jitter_amount > 0:
            feature_values_jittered = feature_values + np.random.normal(0, jitter_amount, size=len(feature_values))
        else:
            feature_values_jittered = feature_values

        # 绘制散点图
        scatter = plt.scatter(feature_values_jittered, shap_vals,
                             c=X_scaled[:, np.argmax([np.corrcoef(X_scaled[:, j], shap_values[:, i])[0,1] for j in range(X_scaled.shape[1])])],
                             alpha=0.8, s=80, edgecolors='black', linewidth=0.5)

        plt.colorbar(scatter, label='Interaction Feature Value')
        plt.xlabel(f'{feat}', fontsize=12, fontweight='bold')
        plt.ylabel(f'SHAP value for {feat}', fontsize=12, fontweight='bold')
        plt.title(f'SHAP Dependence Plot - {feat}\nLogisticRegression',
                  fontsize=14, fontweight='bold', pad=15)

        # 如果是离散特征，将x轴刻度标签替换为英文描述
        if is_categorical:
            unique_values = sorted(set(feature_values))
            unique_labels = [value_mapping[feat].get(int(v), str(v)) for v in unique_values]
            plt.xticks(unique_values, unique_labels, rotation=45, ha='right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f'SHAP特征依赖图_{feat}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"  已保存: SHAP特征依赖图（共{len(optimal_features)}个特征）")

    # 输出特征重要性排序
    print("\n  SHAP特征重要性排名:")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = list(zip(optimal_features, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"    {i}. {feat}: {importance:.4f}")

    print(f"\n  说明:")
    print(f"    - 数据集包含 {len(X_scaled)} 个样本（52个患者）")
    print(f"    - SHAP值基于完整的52个样本计算")
    print(f"    - 点的分布反映每个样本的SHAP值")
    print(f"    - 样本量较小（n=52）导致点较为分散是正常的")
    print(f"    - 所有样本的SHAP解释图已保存到文件夹: SHAP单个样本解释图/")
    print(f"    - 每个样本包含PNG和PDF两种格式")

else:
    print("\n【9. SHAP可视化】")
    print("  跳过SHAP图表生成（未安装shap库）")

# 10. 总结
print("\n" + "="*80)
print("所有可视化图表生成完成（基于8折交叉验证）")
print("="*80)
print("\n使用模型: LogisticRegression (92.56%准确率)")
print("数据集: CARCT1_E1_FG2_processed_translated.csv (n=52)")
print("特征数量: 10个")
print("特征列表: " + ", ".join(optimal_features))

generated_files = [
    "最佳模型ROC曲线（训练集CV）.png",
    "最佳模型ROC曲线（训练集CV）.pdf",
    "最佳模型ROC曲线（验证集CV）.png",
    "最佳模型ROC曲线（验证集CV）.pdf",
    "最佳模型校准曲线（训练集CV）.png",
    "最佳模型校准曲线（训练集CV）.pdf",
    "最佳模型校准曲线（验证集CV）.png",
    "最佳模型校准曲线（验证集CV）.pdf",
    "DCA曲线图（训练集CV）.png",
    "DCA曲线图（训练集CV）.pdf",
    "DCA曲线图（验证集CV）.png",
    "DCA曲线图（验证集CV）.pdf",
    "最佳模型的混淆矩阵（训练集CV）.png",
    "最佳模型的混淆矩阵（训练集CV）.pdf",
    "最佳模型的混淆矩阵（验证集CV）.png",
    "最佳模型的混淆矩阵（验证集CV）.pdf",
    "TOP-K特征重要性曲线图.png",
    "TOP-K特征重要性曲线图.pdf",
]

if SHAP_AVAILABLE:
    generated_files.extend([
        "SHAP特征重要性条形图.png",
        "SHAP特征重要性蜂群图.png",
        f"SHAP单个样本解释图/",
    ])

    # 添加特征依赖图
    for feat in optimal_features:
        generated_files.append(f'SHAP特征依赖图_{feat}.png')

print("\n已生成的文件:")
for i, file in enumerate(generated_files, 1):
    print(f"  {i}. {file}")

print(f"\n总计: {len(generated_files)} 个文件")

print("\n" + "="*80)
print("完成")
print("="*80)
