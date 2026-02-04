# FCR 癌症复发恐惧预测模型

基于 LogisticRegression 的机器学习模型，用于预测患者的癌症复发恐惧（FCR）水平。

## 模型信息

- 模型类型：LogisticRegression
- 准确率：92.56% (8折交叉验证)
- 标准差：±7.48%
- 特征数量：10个

## 在线体验

点击下方链接访问应用：
[Streamlit Cloud 链接待添加]

## 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run app.py
```

## 特征说明

1. **GAD7_0** - GAD-7焦虑评分（0-21分）
2. **TCSQ_NC** - 积极应对方式得分 (TCSQ_NC)（0-100分）
3. **Age** - 年龄（29-66岁）
4. **Residence** - 居住地（0=城市，1=农村）
5. **Education** - 教育程度（0=小学及以下，1=初中，2=高中及以上）
6. **Has_Partner** - 是否有伴侣（0=无，1=有）
7. **Relationship_with_Family** - 与家人关系（1-5分：很差-很好）
8. **Family_Social_Emotional_Support** - 家庭社会情感支持（1-5分：很少-很多）
9. **Perceived_Severity_of_Condition** - 感知疾病严重程度（1-5分：非常轻微-非常严重）
10. **Life_Economic_Stress** - 生活经济压力（1-5分：无压力-很大压力）

## 部署说明

本项目已部署至 Streamlit Cloud，可通过以下方式部署：

### Streamlit Cloud 部署步骤

1. 将代码上传到 GitHub 仓库
2. 访问 [Streamlit Cloud](https://share.streamlit.io)
3. 点击 "New app"
4. 连接你的 GitHub 仓库
5. 配置如下：
   - Repository: 你的仓库名称
   - Branch: main
   - Main file path: app.py
6. 点击 "Deploy"

## 技术栈

- Python 3.x
- Streamlit
- scikit-learn
- pandas
- numpy

## 许可证

MIT License
