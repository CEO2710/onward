
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt

# 定义自变量和因变量
X = df.drop('Unplanned reoperation', axis=1)
y = df['Unplanned reoperation']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型列表
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# 训练和评估模型
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

# 找出最佳模型
best_model_name = max(results, key=lambda x: results[x]['F1-score'])
best_model = models[best_model_name]

# 使用 SHAP 解释最佳模型（假设最佳模型是 Logistic Regression）
if best_model_name == 'Logistic Regression':
    explainer = shap.LinearExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 绘制 SHAP 汇总条形图
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title('SHAP Summary Bar Plot')
    plt.show()

    # 使用 SHAP 平均值评估每个特征对模型的贡献，按降序显示
    feature_importance = pd.DataFrame(
        {'Feature': X.columns, 'SHAP Importance': np.abs(shap_values).mean(axis=0)}
    ).sort_values(by='SHAP Importance', ascending=False)
    print(feature_importance)
