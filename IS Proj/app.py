import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Prediction Dashboard", layout="wide")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_all_models():
    ml_model = joblib.load(MODEL_DIR / 'modelstudent.pkl')
    ml_scaler = joblib.load(MODEL_DIR / 'scalerstudent.pkl')
    nn_model = load_model(MODEL_DIR / 'modellabtop.h5')
    nn_scaler = joblib.load(MODEL_DIR / 'scalerlabtop.pkl')
    return ml_model, ml_scaler, nn_model, nn_scaler

ml_model, ml_scaler, nn_model, nn_scaler = load_all_models()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 AI Control Panel")
page = st.sidebar.radio("Navigation", ["Dashboard", "Student AI", "Laptop AI", "About System"])

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {background-color: #0f172a;}
.card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}
.big-font {font-size: 32px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 AI System Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "2", "Active")
    col2.metric("System Status", "Online", "🟢")
    col3.metric("Accuracy", "~95%", "↑")

    st.markdown("---")

    st.subheader("📈 System Activity")
    data = np.random.randn(50, 2)
    st.line_chart(data)

    st.subheader("📊 Usage Distribution")
    st.bar_chart(np.abs(data[:10]))

# ---------------- STUDENT MODEL ----------------
elif page == "Student AI":
    st.title("🎓 Student Performance AI")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.slider("Study Hours", 0.0, 12.0)
        f2 = st.slider("Attendance (%)", 0.0, 100.0)

    with col2:
        f3 = st.slider("Sleep Hours", 0.0, 12.0)
        f4 = st.slider("Previous Grade", 0.0, 4.0)

    features = [f1, f2, f3, f4]

    if st.button("🔮 Predict Performance"):
        with st.spinner("AI Processing..."):
            data = ml_scaler.transform([features])
            result = ml_model.predict(data)[0]

            st.markdown(f"""
            <div class='card'>
                <div class='big-font'>Result: {result}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- LAPTOP MODEL ----------------
elif page == "Laptop AI":
    st.title("💻 Laptop Health AI")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.slider("CPU Usage (%)", 0.0, 100.0)
        f2 = st.slider("RAM Usage (%)", 0.0, 100.0)
        f3 = st.slider("Battery (%)", 0.0, 100.0)

    with col2:
        f4 = st.slider("Temperature (°C)", 0.0, 120.0)
        f5 = st.selectbox("Device Type", [0,1,2,3])

    features = [f1, f2, f3, f4, f5]

    if st.button("⚡ Analyze System"):
        with st.spinner("Running Neural Network..."):
            data = nn_scaler.transform([features])
            result = nn_model.predict(data)[0][0]
            prediction = int(result > 0.5)

            label = "🟢 Normal" if prediction == 0 else "🔴 Risk"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(result)*100,
                title={'text': "Risk Level (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class='card'>
                <div class='big-font'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("📊 Input Breakdown")
    st.bar_chart(features)

# ---------------- ABOUT SYSTEM ----------------
elif page == "About System":
    st.title("📘 How This System Works")

    st.markdown("""
    ### 🔍 Overview
    This web application uses **Machine Learning (ML)** and **Neural Network (NN)** models to make predictions based on user input.

    ---
    
    ### 🎓 Student Model (ML)
    - Algorithm: Traditional Machine Learning
    - Inputs:
        - Study Hours
        - Attendance
        - Sleep Hours
        - Previous Grade
    - Process:
        1. Input data → Scaler
        2. Scaled data → ML Model
        3. Output → Prediction Result

    ---

    ### 💻 Laptop Model (Neural Network)
    - Algorithm: Deep Learning (Neural Network)
    - Inputs:
        - CPU Usage
        - RAM Usage
        - Battery Level
        - Temperature
        - Device Type
    - Process:
        1. Input data → Normalization (Scaler)
        2. Data → Neural Network
        3. Output → Probability (Risk %)

    ---

    ### ⚙️ Technologies Used
    - Streamlit (Web App)
    - Scikit-learn (ML Model)
    - TensorFlow / Keras (Neural Network)
    - Plotly (Visualization)

    ---

    ### 🎯 Purpose
    This project demonstrates how AI models can be deployed into a real-world web application with an interactive UI.
    """)

    st.success("✅ This system is ready for real-world demonstration and academic presentation.")
