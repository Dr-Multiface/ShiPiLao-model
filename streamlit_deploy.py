import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="视疲劳中医辨证诊断系统",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 样式美化
st.markdown("""
    <style>
        .stButton>button {font-size: 16px; padding: 8px 20px; width: 100%; border-radius: 8px;}
        .stRadio > div {font-size: 14px; padding: 4px 0;}
        .stDataFrame {font-size: 12px;}
        h1 {font-size: 22px; text-align: center; color: #165DFF;}
        h2 {font-size: 18px; margin-top: 20px;}
        .stInfo {padding: 10px; font-size: 14px;}
        .stSuccess {background-color: #f0f9ff; border-left: 4px solid #165DFF;}
    </style>
""", unsafe_allow_html=True)

# 网页标题
st.title("✅ 视疲劳中医辨证诊断系统")
st.markdown("<p style='text-align:center; color:#666;'>26项症状筛查 | LR逻辑回归模型 | 精准辨证分型</p>", unsafe_allow_html=True)
st.divider()

# 加载模型和标准化器
@st.cache_resource
def load_lr_model():
    try:
        lr_model = joblib.load("saved_models/ship_lr_model.pkl")
        scaler = joblib.load("saved_models/ship_scaler.pkl")
        st.success("✅ 模型加载成功！可开始辨证预测")
        return lr_model, scaler
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.warning("⚠️ 请先运行 train_lr_model.py 训练并保存模型！")
        return None, None

model, scaler = load_lr_model()

# 数据预处理函数
def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    df_scaled = scaler.transform(df)
    return df_scaled

# 核心映射：数字标签 → 中医证型
zhengxing_dict = {
    1: "肝气郁结证",
    2: "气血两虚证",
    3: "脾虚气弱证",
    4: "肝肾阴虚证"
}

# ===================== 批量辨证预测 【保留原有功能，无修改】 =====================
st.subheader("📁 批量辨证 - 上传症状特征CSV文件")
st.info("✅ 要求：CSV格式，列顺序【序号列+26特征列】，无标签列，自动跳过序号列")
uploaded_file = st.file_uploader("点击上传CSV文件", type=["csv"])

if uploaded_file is not None and model is not None and scaler is not None:
    df_input = pd.read_csv(uploaded_file)
    st.info(f"✅ 上传成功 | 数据量：{df_input.shape[0]} 条 | 总列数：{df_input.shape[1]} 列")
    st.dataframe(df_input.head(3), use_container_width=True)
    
    df_feature = df_input.iloc[:, 1:]
    with st.spinner("🔍 数据预处理中...辨证预测中..."):
        df_processed = preprocess_data(df_feature)
        pred_label = model.predict(df_processed)
        pred_label_origin = pred_label + 1
        pred_zhengxing = [zhengxing_dict[num] for num in pred_label_origin]
    
    df_input["中医辨证结果"] = pred_zhengxing
    st.success("✅ 批量辨证预测完成！")
    st.dataframe(df_input, use_container_width=True)
    
    # 下载结果，解决中文乱码
    csv_result = df_input.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 下载辨证结果CSV",
        data=csv_result,
        file_name="中医辨证预测结果.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()

# ===================== 单条辨证预测 - 核心修改：数值框→是/否选择题 =====================
st.subheader("✏️ 单条辨证 - 症状筛查问卷（请选择「是」或「否」）")
st.info("✅ 请根据自身情况选择，完成后点击预测即可获取辨证结果")

feature_values = []
cols = st.columns(2)  # 改为2列布局，更适合选择题展示（4列太挤）

# ✅ 核心修改1：26个特征→通俗化问题描述（大众易懂）+ 是/否单选框
# 格式：["通俗问题描述", 原始特征名]
popular_feature_names = [
    ["你是否经常感觉眼睛干涩？", "眼干"],
    ["你是否经常感觉眼睛发酸？", "眼酸"],
    ["你是否经常感觉眼睛胀痛？", "眼胀痛"],
    ["你是否经常感觉腹部胀闷不适？", "脘腹作胀"],
    ["你是否经常感到心情抑郁、不开心？", "自觉抑郁"],
    ["你是否容易急躁、发脾气？", "急躁易怒"],
    ["你是否经常不自觉地叹气？", "喜叹息"],
    ["你是否经常感觉眼睛疲倦、不想睁眼？", "眼疲倦"],
    ["你是否感觉最近视力有所下降？", "视力减退"],
    ["你是否容易出汗（不热也出汗）？", "感到自汗"],
    ["你是否经常感觉四肢无力、懒得动？", "四肢倦怠"],
    ["你是否经常失眠、多梦？", "失眠多梦"],
    ["你是否感觉上眼皮沉重、抬不起来？", "上眼皮沉重"],
    ["你是否有眼睑不自主跳动的情况？", "眼睑痉挛"],
    ["你是否饭量减少、不想吃饭？", "纳少"],
    ["你是否感觉记忆力变差、容易忘事？", "记忆力差"],
    ["你是否经常注意力不集中、容易走神？", "注意力不集中"],
    ["你是否经常感觉精神疲惫、浑身乏力？", "神疲乏力"],
    ["你是否不想说话、说话声音小？", "少气懒言"],
    ["你是否大便稀烂、不成形？", "大便稀溏"],
    ["你是否看东西模糊不清？", "视力模糊"],
    ["你是否经常头晕或头痛？", "头晕或头痛"],
    ["你是否有耳鸣的情况？", "耳鸣"],
    ["你是否经常感觉腰部和膝盖酸软无力？", "腰膝酸软"],
    ["你是否感觉手心、脚心、心口发热？", "五心烦热"],
    ["你是否经常感觉咽喉和嘴巴干燥？", "咽干口干"]
]

# 循环渲染单选框时的赋值逻辑修改
for i in range(len(popular_feature_names)):
    with cols[i % 2]:
        question = popular_feature_names[i][0]
        choice = st.radio(
            question,
            options=["是", "否"],
            key=f"feat_{i}",
            horizontal=True,
            index=None  # 无默认选项
        )
        # 兼容未选择的情况，未选择时赋值为None
        val = 1.0 if choice == "是" else (0.0 if choice == "否" else None)
        feature_values.append(val)

# 单条预测按钮（增加必填项校验）
if st.button("🚀 立即辨证预测", type="primary", use_container_width=True) and model is not None and scaler is not None:
    # 校验是否所有问题都已选择（无默认值时，未选择会返回None）
    if None in feature_values:
        st.error("❌ 请完成所有26个症状问题的「是/否」选择后再预测！")
    else:
        with st.spinner("🔮 正在辨证预测中..."):
            X_single = np.array(feature_values).reshape(1, -1)
            X_single_scaled = scaler.transform(X_single)
            pred = model.predict(X_single_scaled)[0]
            pred_origin = pred + 1
            final_result = zhengxing_dict[pred_origin]
        
        st.divider()
        st.markdown(f"""
            <div style='text-align:center; padding:20px; border-radius:12px; background:#f0f9ff; border:1px solid #91caff;'>
                <h2 style='color:#165DFF; margin:0; font-weight:bold;'>辨证结果：{final_result}</h2>
            </div>
        """, unsafe_allow_html=True)

# 底部说明
st.divider()
st.caption("✅ 辨证分型：肝气郁结证 | 气血两虚证 | 脾虚气弱证 | 肝肾阴虚证")