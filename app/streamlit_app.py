import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Custom Gradient Background & CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #A8C0FF 0%, #3f2b96 100%);
}
.st-emotion-cache-18ni7ap, .st-emotion-cache-6qob1r {
    background: transparent;
}
[data-testid="stSidebar"] {
    background-image: linear-gradient(135deg, #5EFCE8 10%, #7367F0 100%);
}
.st-emotion-cache-1v0mbdj {
    border-radius: 18px;
}
.big-font {
    font-size:2.2em !important;
    font-weight:600;
    color:#383e56;
}
.metric-card {
    border-radius: 16px;
    background: #fff2;
    padding: 22px;
    margin: 8px 0;
    box-shadow: 0 4px 24px #3f2b9622;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# --- Config and Sidebar ---
st.set_page_config(page_title="🧠 Wellness Predictor", layout="wide", page_icon="🧠")
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mental-health.png", width=100)
    st.title("Welcome!")
    st.markdown("""
    **Mental Health Predictor**  
    Gain insights about your well-being 🌱  
    _Not a substitute for professional advice_
    """)
    st.write("— Powered by Machine Learning & Streamlit —")
    st.divider()
    st.markdown("#### Navigation")
    page = st.radio("Jump to", ["🌟 Home", "📊 Analytics", "💡 Tips", "🗣️ Feedback"], label_visibility='collapsed')

MODEL_PATH="models/final_model.pkl"
SCALER_PATH="models/scaler.pkl"
LE_PATH="models/label_encoder.pkl"

def load_resource(path):
    return joblib.load(path) if os.path.exists(path) else None
model = load_resource(MODEL_PATH)
scaler = load_resource(SCALER_PATH)
le = load_resource(LE_PATH)

if not all([model, scaler, le]):
    st.error("Model or encoder not found. Please run the training first.")
    st.stop()

# --- Page Content ---
if page == "🌟 Home":
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("<div class='big-font'>🌈 How are you today?</div>", unsafe_allow_html=True)
        st.write("#### Enter your daily habits for a personalized stress level analysis.")
        with st.form("prediction_form"):
            sleep = st.slider("🛌 Sleep hours", 0, 12, 7)
            study = st.slider("📚 Study/work hours", 0, 12, 4)
            screen = st.slider("📱 Screen time (hrs)", 0, 16, 4)
            social = st.slider("🧑‍🤝‍🧑 Social activity (0=low, 5=high)", 0, 5, 3)
            diet = st.slider("🍎 Diet quality (1=poor, 5=excellent)", 1, 5, 3)
            exercise = st.radio("🏃 Exercise regularly?", ["Yes", "No"])
            exercise_val = 1 if exercise == "Yes" else 0
            submitted = st.form_submit_button("✨ Predict My Wellness ✨")
        if submitted:
            x = np.array([[sleep, study, screen, social, diet, exercise_val]])
            try:
                x_scaled = scaler.transform(x)
                pred = model.predict(x_scaled)
                label = le.inverse_transform(pred)[0]
                color_map = {"Low": "#58ed91", "Medium": "#FFD700", "High": "#fc3d39"}
                st.markdown(
                    f"""
                    <div class="metric-card" style="background:{color_map.get(label,'#dedede')};">
                        <span class='big-font'>{label} Stress</span><br>
                        <span style='font-size:1.2em;'>based on your lifestyle</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.info("If you feel stressed, consider reaching out to someone you trust or a professional.")
                st.write("##### 📊 Your Inputs vs Recommended:")
                st.dataframe(pd.DataFrame({
                    "Habit": ["Sleep (hrs)", "Screen time (hrs)", "Exercise", "Diet"],
                    "Your Value": [sleep, screen, "Yes" if exercise_val else "No", diet],
                    "Recommended": ["7-9", "<4", "Yes", "≥4"]
                }))
            except Exception as e:
                st.error(f"Prediction error: {e}")
        st.markdown("---")
        with st.expander("💡 Why These Habits Matter?", expanded=False):
            st.write("""
            - **Sleep:** Essential for brain and body recovery.
            - **Screen Time:** Too much affects mood and sleep.
            - **Exercise:** Boosts endorphins.
            - **Diet:** Fuels your mind and stabilizes mood.
            - **Social:** Healthy interaction reduces anxiety.
            """)

    with c2:
        st.image("https://img.freepik.com/free-vector/mental-health-concept-illustration_114360-2037.jpg")
        st.markdown("""
        <div style='padding:18px; background:rgba(255,255,255,0.5); border-radius:12px; margin-top:15px;'>
        <b>Need Help?</b><br>
        <span style='font-size:1.1em;'>If you're in crisis, talk to someone you trust or reach your local helpline.</span>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Analytics":
    st.header("📈 Insights & Feature Importance")
    st.write("#### See what affects your stress most (model-based feature importances):")
    feat_labels = ["Sleep", "Study", "Screen", "Social", "Diet", "Exercise"]
    try:
        importances = model.feature_importances_
        df_feat = pd.DataFrame({"Feature": feat_labels, "Importance": importances})
        st.bar_chart(df_feat.set_index("Feature"))
    except Exception:
        st.warning("Model does not support feature_importances_")

elif page == "💡 Tips":
    st.header("🦋 Wellness Tips & Motivation")
    st.markdown("""
        - Try keeping a **gratitude journal** for daily reflection.
        - Practice **deep breathing** or short mindful walks.
        - Reduce notification overload & "digital detox".
        - Balanced meals and hydration help battle stress.
        - Connect with friends, family, or support groups regularly.
    """)
    st.image("https://img.freepik.com/free-vector/wellness-concept-illustration_114360-6697.jpg")

elif page == "🗣️ Feedback":
    st.header("🤔 Share Your Feedback")
    st.write("How was your experience?")
    st.slider("App Rating", 1, 5, 4)
    st.text_area("What features would you like? Or any suggestions?")
    st.button("📤 Submit Feedback")

st.markdown("""
    ---
    <center>
    <small>Made with 💙 for awareness. © 2025.</small>
    </center>
""", unsafe_allow_html=True)
