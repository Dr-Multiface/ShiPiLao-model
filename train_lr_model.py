# 导入所需库
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告

# ===================== 1. 加载数据集 (VSCode本地路径，无需修改) =====================
data_path = "ShiPiLaoMaXiaoTian20251113.csv"
df = pd.read_csv(data_path)

# 数据基本信息查看
print("数据集基本信息：")
print(f"数据集形状: {df.shape}")
print("\n前5行数据：")
print(df.head())
print("\n缺失值统计：")
print(df.isnull().sum())

# ===================== 2. 数据预处理 【核心精准匹配你的数据结构，重中之重】 =====================
# ✅ 严格匹配：第1列=序号(剔除)，第2-27列=26个特征列，第28列=标签列
X = df.iloc[:, 1:-1]  # 关键修正：取第2列到倒数第2列 → 刚好26个特征列，剔除序号列和标签列
y = df.iloc[:, -1]    # 最后1列：标签列 Species (取值1,2,3,4)

# 标签修正 1-4 → 0-3 （LR多分类模型必须从0开始的连续整数）
y = y - 1
print(f"\n修正后标签取值: {sorted(y.unique())}") # 输出 [0,1,2,3] 确认修正成功

# LR模型对特征尺度极度敏感，标准化【必须保留，保证模型效果】
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集（参数不变，保证数据分布一致）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}") # 正确输出 (191,26) (48,26)

# ===================== 3. 定义+训练LR多分类模型 【无报错，兼容所有sklearn版本】 =====================
# ✅ 已删除冲突的multi_class参数，适配低版本sklearn，核心调参全部保留（最优参数不变）
lr_model = LogisticRegression(
    solver='lbfgs',
    C=0.1,                      # 正则化强度，抗过拟合最优值，重中之重
    penalty='l2',              # L2正则，防止过拟合
    max_iter=2000,             # 确保模型收敛完全
    class_weight='balanced',   # 平衡类别权重，解决样本不均衡问题
    random_state=42            # 固定随机种子，结果可复现
)
lr_model.fit(X_train, y_train)

# ===================== 4. 5折交叉验证 + 模型评估 (完全保留，无修改) =====================
print("\n===== 5折分层交叉验证开始 =====")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_model, X_scaled, y, cv=cv, scoring='accuracy')
print(f"5折交叉验证各折准确率: {[round(score, 4) for score in cv_scores]}")
print(f"5折交叉验证平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 训练集/测试集准确率 + 详细分类报告
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\n训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print("\n===== 详细分类报告 =====")
print(classification_report(y_test, y_test_pred))

# ===================== 5. 保存LR模型 + 标准化器 【本地保存，VSCode专用，无需修改】 =====================
save_dir = "saved_models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, "ship_lr_model.pkl")
scaler_path = os.path.join(save_dir, "ship_scaler.pkl")

joblib.dump(lr_model, model_path)
joblib.dump(scaler, scaler_path)

print("\n✅ 模型保存成功！生成的文件路径（当前目录下）：")
print(f"1. 逻辑回归模型: {model_path}")
print(f"2. 特征标准化器: {scaler_path}")
print("保存完成，可直接运行Streamlit部署代码！")