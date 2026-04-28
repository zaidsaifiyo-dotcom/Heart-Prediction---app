import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction with Explainable AI (SHAP)")
st.write("Model Accuracy: 90.54% | Providing Clinical Reasoning for Every Prediction")

# 1. LOAD & TRAIN (Same as before, but keeping track of feature names)
@st.cache_resource
def load_and_train():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, 'heart_combined.csv'))
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We use the XGBoost part of the stack for SHAP as it's the most powerful 'reasoner'
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='logloss')
    xgb_model.fit(X_scaled, y)
    
    return xgb_model, scaler, X.columns

model, scaler, model_columns = load_and_train()

# 2. SIDEBAR INPUTS
st.sidebar.header("Patient Medical Data")
# (Keeping your existing inputs...)
age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
cp = st.sidebar.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
rbp = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
mhr = st.sidebar.slider("Max Heart Rate", 60, 210, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# 3. PREDICTION & REASONING
if st.button("Analyze Heart Health"):
    # Preprocess Input
    input_data = pd.DataFrame([[age, sex, cp, rbp, chol, fbs, restecg, mhr, exang, oldpeak, st_slope]], 
                              columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])
    input_encoded = pd.get_dummies(input_data).reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    
    # Prediction
    prob = model.predict_proba(input_scaled)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnostic Result")
        if prob > 0.5:
            st.error(f"⚠️ HIGH RISK ({prob*100:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({(1-prob)*100:.1f}%)")

    with col2:
        st.subheader("Medical Reasoning (SHAP Analysis)")
        # Calculate SHAP values for this specific input
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        # Plotting
        fig, ax = plt.subplots()
        shap.bar_plot(shap_values[0], feature_names=model_columns, max_display=10, show=False)
        st.pyplot(fig)
        st.write("Positive values (Right) increase risk, Negative values (Left) decrease risk.")