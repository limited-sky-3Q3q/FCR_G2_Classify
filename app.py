"""
FCR 预测模型 Web 应用
基于 LogisticRegression (92.56% 准确率)
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 页面设置
st.set_page_config(
    page_title="FCR 预测模型",
    page_icon="🧠",
    layout="wide"
)

# 标题
st.title("🧠 FCR 分类预测模型")
st.markdown("""
基于 LogisticRegression 模型（准确率: 92.56%）

请输入以下10个特征值，系统将预测 FCR_G2 分类结果。
""")

# 加载模型
@st.cache_resource
def load_model():
    model_path = '各模型最优参数（可复现）\\fcr_web_model.pkl'

    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)

        return (
            model_info['model'],
            model_info['scaler'],
            model_info['imputer_statistics'],  # 返回统计量列表
            model_info['optimal_features'],
            model_info.get('feature_descriptions', {})
        )
    except FileNotFoundError:
        st.error(f"模型文件不存在: {model_path}")
        st.info("请先运行 '1各模型最优参数（可复现）\\生成Web模型.py' 生成模型文件")
        st.stop()
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        st.stop()

model, scaler, imputer_stats, optimal_features, feature_descriptions = load_model()

# 特征描述（使用原始数据显示）
if not feature_descriptions:
    feature_descriptions = {
        'GAD7_0': 'GAD-7焦虑评分',
        'TCSQ_NC': '积极应对方式得分 (TCSQ_NC)',
        'Age': '年龄',
        'Residence': '居住地',
        'Education': '教育程度',
        'Has_Partner': '是否有伴侣',
        'Relationship_with_Family': '与家人关系',
        'Family_Social_Emotional_Support': '家庭社会情感支持',
        'Perceived_Severity_of_Condition': '感知疾病严重程度',
        'Life_Economic_Stress': '生活经济压力'
    }

# 输入表单
st.subheader("📊 特征输入")

col1, col2 = st.columns(2)

with col1:
    GAD7_0 = st.slider(feature_descriptions['GAD7_0'], min_value=0, max_value=21, value=6, key='GAD7_0')  # 中间值 (0+21)/2
    TCSQ_NC = st.slider(feature_descriptions['TCSQ_NC'], min_value=10, max_value=50, value=18, key='TCSQ_NC') 
    Age = st.slider(feature_descriptions['Age'], min_value=29, max_value=66, value=42, key='Age')  # 中间值 (29+66)/2
    Residence = st.selectbox(feature_descriptions['Residence'], options=[0, 1], format_func=lambda x: "城市" if x == 0 else "农村", key='Residence', index=0)
    Education = st.selectbox(feature_descriptions['Education'], options=[0, 1, 2], format_func=lambda x: ['小学及以下', '初中', '高中及以上'][x], key='Education', index=1)  # 中间值

with col2:
    Has_Partner = st.selectbox(feature_descriptions['Has_Partner'], options=[0, 1], format_func=lambda x: "无" if x == 0 else "有", key='Has_Partner', index=1)
    Relationship_with_Family = st.selectbox(feature_descriptions['Relationship_with_Family'], options=[1, 2, 3, 4, 5], format_func=lambda x: ['很差', '较差', '一般', '较好', '很好'][x-1], key='Relationship_with_Family', index=2)  # 中间值
    Family_Social_Emotional_Support = st.selectbox(feature_descriptions['Family_Social_Emotional_Support'], options=[1, 2, 3, 4, 5], format_func=lambda x: ['很少', '较少', '一般', '较多', '很多'][x-1], key='Family_Social_Emotional_Support', index=2)  # 中间值
    Perceived_Severity_of_Condition = st.selectbox(feature_descriptions['Perceived_Severity_of_Condition'], options=[1, 2, 3, 4, 5], format_func=lambda x: ['非常轻微', '轻微', '中度', '严重', '非常严重'][x-1], key='Perceived_Severity_of_Condition', index=2)  # 中间值
    Life_Economic_Stress = st.selectbox(feature_descriptions['Life_Economic_Stress'], options=[1, 2, 3, 4, 5], format_func=lambda x: ['无压力', '轻微压力', '中度压力', '较大压力', '很大压力'][x-1], key='Life_Economic_Stress', index=2)  # 中间值

# 预测按钮
predict_button = st.button("🔮 进行预测", type="primary", use_container_width=True)

# 预测结果
if predict_button:
    # 构建输入数据
    input_data = pd.DataFrame([{
        'GAD7_0': GAD7_0,
        'TCSQ_NC': TCSQ_NC,
        'Age': Age,
        'Residence': Residence,
        'Education': Education,
        'Has_Partner': Has_Partner,
        'Relationship_with_Family': Relationship_with_Family,
        'Family_Social_Emotional_Support': Family_Social_Emotional_Support,
        'Perceived_Severity_of_Condition': Perceived_Severity_of_Condition,
        'Life_Economic_Stress': Life_Economic_Stress
    }], columns=optimal_features)

    # 数据预处理 - 将1-5分映射回处理后的数值
    input_array = input_data.values
    
    # 特征映射：1-5分 -> 0-4 或 0-3 或 0-2
    # Relationship_with_Family: 1-5 -> 0-4
    input_array[0, optimal_features.index('Relationship_with_Family')] -= 1
    # Family_Social_Emotional_Support: 1-5 -> 0-3 (训练数据范围)
    input_array[0, optimal_features.index('Family_Social_Emotional_Support')] = min(input_array[0, optimal_features.index('Family_Social_Emotional_Support')] - 1, 3)
    # Perceived_Severity_of_Condition: 1-5 -> 0-2 (训练数据范围)
    input_array[0, optimal_features.index('Perceived_Severity_of_Condition')] = min(input_array[0, optimal_features.index('Perceived_Severity_of_Condition')] - 1, 2)
    # Life_Economic_Stress: 1-5 -> 0-3 (训练数据范围)
    input_array[0, optimal_features.index('Life_Economic_Stress')] = min(input_array[0, optimal_features.index('Life_Economic_Stress')] - 1, 3)

    # 简单处理：如果有缺失值用0填充（Web输入不会有缺失值）
    input_array = np.nan_to_num(input_array, nan=0.0)

    # 标准化
    input_scaled = scaler.transform(input_array)

    # 预测
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0, 1]

    # 显示结果
    st.markdown("---")
    st.subheader("🎯 预测结果")

    # 结果卡片
    result_class = 1 if probability >= 0.5 else 0
    
    # 获取两个分类的概率
    prob_class_0 = model.predict_proba(input_scaled)[0, 0]  # FCR_G2=1 的概率
    prob_class_1 = model.predict_proba(input_scaled)[0, 1]  # FCR_G2=2 的概率

    col_result, col_prob = st.columns(2)

    with col_result:
        if result_class == 0:
            st.success("FCR_G2 = 1")
            st.info("低度癌症复发恐惧")
        else:
            st.warning("FCR_G2 = 2")
            st.info("高度癌症复发恐惧")

    with col_prob:
        # 显示对应分类结果的概率
        if result_class == 0:
            display_prob = prob_class_0
            label = "属于 FCR_G2=1 的概率"
        else:
            display_prob = prob_class_1
            label = "属于 FCR_G2=2 的概率"
        
        st.markdown("### 预测概率")
        st.metric(label=label, value=f"{display_prob:.2%}")
        st.progress(display_prob)

    # 特征贡献
    st.markdown("---")
    st.subheader("📈 特征重要性分析")

    coefficients = model.coef_[0]
    feature_contributions = []

    for feat, coef in zip(optimal_features, coefficients):
        contribution = coef * input_data[feat].values[0]
        feature_contributions.append({
            'Feature': feat,
            'Description': feature_descriptions[feat],
            'Coefficient': coef,
            'Value': input_data[feat].values[0],
            'Contribution': contribution
        })

    df_contributions = pd.DataFrame(feature_contributions).sort_values('Contribution', ascending=True)

    # 显示特征贡献
    st.bar_chart(df_contributions.set_index('Feature')['Contribution'])

    # 详细贡献表格
    with st.expander("📋 查看详细特征贡献"):
        st.dataframe(
            df_contributions.style.format({
                'Coefficient': '{:.4f}',
                'Value': '{:.2f}',
                'Contribution': '{:.4f}'
            }),
            use_container_width=True
        )

# 模型信息
st.markdown("---")
with st.expander("ℹ️ 模型详细信息"):
    st.markdown(f"""
    **模型类型**: LogisticRegression
    **准确率**: 92.56% (8折交叉验证)
    **标准差**: ±7.48%

    **最优特征 (10个)**:
    1. GAD7_0 - GAD-7焦虑评分（0-21分）
    2. TCSQ_NC - 积极应对方式得分（10-50分）
    3. Age - 年龄（0-99岁）
    4. Residence - 居住地（0=城市，1=农村）
    5. Education - 教育程度（0=小学及以下，1=初中，2=高中及以上）
    6. Has_Partner - 是否有伴侣（0=无，1=有）
    7. Relationship_with_Family - 与家人关系（1-5分：很差-很好）
    8. Family_Social_Emotional_Support - 家庭社会情感支持（1-5分：很少-很多）
    9. Perceived_Severity_of_Condition - 感知疾病严重程度（1-5分：非常轻微-非常严重）
    10. Life_Economic_Stress - 生活经济压力（1-5分：无压力-很大压力）
    """)
