import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ─── Page config MUST be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="MindPulse · Wellness Predictor",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ─── Design tokens ───────────────────────────────────────────────────────────
# Primary palette  : Glaucous #6082B6 (calming, trustworthy blue)
# Accent palette   : Sage Green #7BA05B (wellness, natural)
# Background dark  : #0F1624  Cards: #141D2E  Border: #1E2D45
# Text primary     : #E2E8F0  Secondary: #94A3B8  Muted: #475569
# Stress low  : #22C55E  medium: #F59E0B  high: #EF4444

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Global reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: #0F1624 !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0C1220 !important;
    border-right: 1px solid #1E2D45 !important;
}
[data-testid="stSidebar"] * {
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E2E8F0 !important;
}

/* ── Radio nav pills ── */
[data-testid="stSidebar"] .stRadio > label {
    display: none;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 10px 14px !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    color: #94A3B8 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(96,130,182,0.1) !important;
    border-color: rgba(96,130,182,0.25) !important;
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: rgba(96,130,182,0.18) !important;
    border-color: rgba(96,130,182,0.4) !important;
    color: #E2E8F0 !important;
}

/* ── Main text colors ── */
h1, h2, h3, h4, h5, h6 {
    color: #E2E8F0 !important;
}
p, li, span, label, div {
    color: #CBD5E1;
}

/* ── Glass card utility ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: box-shadow 0.2s;
}
.glass-card:hover {
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

/* ── Hero title ── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.25;
    background: linear-gradient(135deg, #E2E8F0 30%, #6082B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94A3B8 !important;
    line-height: 1.6;
    margin-bottom: 24px;
}
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6082B6 !important;
    margin-bottom: 6px;
}

/* ── Sliders ── */
.stSlider > label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #6082B6 !important;
    border-color: #6082B6 !important;
}

/* ── Radio buttons (form) ── */
.stRadio > label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stRadio div[role="radiogroup"] label span {
    color: #CBD5E1 !important;
}

/* ── Submit / primary button ── */
.stFormSubmitButton > button,
button[kind="primary"] {
    background: linear-gradient(135deg, #6082B6 0%, #4A6CA0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(96,130,182,0.35) !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
}
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 28px rgba(96,130,182,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary buttons ── */
button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
}

/* ── Dataframe / table ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.02) !important;
}

/* ── Expander ── */
details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
summary {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.07) !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    background: rgba(96,130,182,0.12) !important;
}

/* ── Select slider ── */
.stSelectSlider > label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Multiselect ── */
.stMultiSelect > label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stMultiSelect [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
.stMultiSelect span {
    color: #E2E8F0 !important;
}

/* ── Text area ── */
.stTextArea > label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea::placeholder {
    color: #475569 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2D4265; border-radius: 3px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─── Model loading ────────────────────────────────────────────────────────────
MODEL_PATH  = "models/final_model.pkl"
SCALER_PATH = "models/scaler.pkl"
LE_PATH     = "models/label_encoder.pkl"

@st.cache_resource
def load_artifacts():
    m  = joblib.load(MODEL_PATH)  if os.path.exists(MODEL_PATH)  else None
    sc = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    le = joblib.load(LE_PATH)     if os.path.exists(LE_PATH)     else None
    return m, sc, le

model, scaler, le = load_artifacts()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:16px 0 22px; text-align:center;">
            <div style="font-size:2.8rem; line-height:1;">🧠</div>
            <div style="font-size:1.25rem; font-weight:700; color:#E2E8F0; margin-top:10px;">
                MindPulse
            </div>
            <div style="font-size:0.75rem; color:#475569; margin-top:3px; letter-spacing:0.05em;">
                WELLNESS PREDICTOR
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("<div class='section-label' style='padding:0 4px;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🌟 Home", "📊 Analytics", "💡 Tips", "💬 AI Chat", "🗣️ Feedback"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        """
        <div style="background:rgba(96,130,182,0.1); border:1px solid rgba(96,130,182,0.2);
                    border-radius:12px; padding:14px; margin-top:4px;">
            <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.1em;
                        text-transform:uppercase; color:#6082B6; margin-bottom:6px;">
                Disclaimer
            </div>
            <div style="font-size:0.78rem; color:#64748B; line-height:1.55;">
                Not a substitute for professional mental health advice or diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="margin-top:24px; text-align:center;">
            <div style="font-size:0.68rem; color:#334155;">
                Powered by ML · Made with 💙
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Guard: model files ───────────────────────────────────────────────────────
if not all([model, scaler, le]):
    st.error(
        "⚠️ Model files not found. "
        "Please run `python src/train_model.py` from the project root first."
    )
    st.stop()

# ─── Chart helpers ────────────────────────────────────────────────────────────

def make_gauge(label: str) -> go.Figure:
    color_map = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}
    value_map = {"Low": 18,        "Medium": 55,         "High": 88}
    color = color_map.get(label, "#6082B6")
    value = value_map.get(label, 50)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"color": color, "size": 40, "family": "Inter"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#1E2D45",
                    "tickfont": {"color": "#475569", "size": 10},
                },
                "bar": {"color": color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "rgba(34,197,94,0.12)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.12)"},
                    {"range": [65,100], "color": "rgba(239,68,68,0.12)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.78,
                    "value": value,
                },
            },
            title={
                "text": f"<b style='color:{color}'>{label} Stress</b>",
                "font": {"color": "#E2E8F0", "size": 17, "family": "Inter"},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        height=230,
        font={"family": "Inter"},
    )
    return fig


def make_radar(sleep, study, screen, social, diet, exercise_val) -> go.Figure:
    cats = ["Sleep", "Study", "Screen<br>(inverted)", "Social", "Diet", "Exercise"]
    user = [
        min(sleep / 9, 1.0),
        min(study / 8, 1.0),
        max(1 - screen / 16, 0.0),
        social / 5,
        (diet - 1) / 4,
        float(exercise_val),
    ]
    ideal = [1.0, 0.5, 0.75, 0.8, 1.0, 1.0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ideal, theta=cats, fill="toself", name="Ideal",
        line=dict(color="#6082B6", width=2),
        fillcolor="rgba(96,130,182,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=user, theta=cats, fill="toself", name="You",
        line=dict(color="#7BA05B", width=2),
        fillcolor="rgba(123,160,91,0.18)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#1E2D45",
                tickfont={"color": "#475569", "size": 9},
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(gridcolor="#1E2D45", tickfont={"color": "#94A3B8", "size": 11}),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font={"color": "#94A3B8", "size": 11}, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=40, b=40),
        height=310,
        font={"family": "Inter"},
    )
    return fig


def make_feature_bar(feat_labels, importances) -> go.Figure:
    df = (
        pd.DataFrame({"Feature": feat_labels, "Importance": importances})
        .sort_values("Importance", ascending=True)
    )
    max_idx = df["Importance"].idxmax()
    colors = [
        "#6082B6" if i == max_idx else "#3B5780"
        for i in df.index
    ]

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=11, family="Inter"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1E2D45", color="#475569", zeroline=False, showgrid=True),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#94A3B8", tickfont=dict(size=12)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=300,
        font={"family": "Inter", "color": "#94A3B8"},
        showlegend=False,
    )
    return fig


def make_donut(feat_labels, importances) -> go.Figure:
    palette = ["#6082B6", "#7BA05B", "#5B9C9F", "#9F5B7B", "#B79562", "#5B7B9F"]
    fig = go.Figure(go.Pie(
        labels=feat_labels,
        values=importances,
        hole=0.62,
        marker=dict(colors=palette, line=dict(color="#0F1624", width=2)),
        textfont=dict(color="#E2E8F0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font={"color": "#94A3B8", "size": 11}, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=280,
        font={"family": "Inter"},
        annotations=[{
            "text": "Features", "x": 0.5, "y": 0.5,
            "font_size": 13, "showarrow": False, "font_color": "#94A3B8",
        }],
    )
    return fig


# ─── Utility: hex → rgb tuple string ─────────────────────────────────────────
def hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🌟 Home":
    st.markdown("<div class='section-label'>Daily Wellness Check</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>How are you doing today?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Enter your daily lifestyle habits below and our ML model will assess your current stress level in real time.</div>",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 📝 Your Daily Habits")
        with st.form("wellness_form"):
            sleep   = st.slider("🛌 Sleep hours",            0, 12,  7, help="Recommended: 7–9 hours per night")
            study   = st.slider("📚 Study / Work hours",     0, 12,  4, help="Balance work with adequate breaks")
            screen  = st.slider("📱 Screen time (hrs)",      0, 16,  4, help="Recommended: under 4 hours")
            social  = st.slider("🧑‍🤝‍🧑 Social activity",       0,  5,  3, help="0 = isolated · 5 = very active")
            diet    = st.slider("🍎 Diet quality",           1,  5,  3, help="1 = poor · 5 = excellent")
            exercise = st.radio("🏃 Exercise regularly?", ["Yes", "No"], horizontal=True)
            exercise_val = 1 if exercise == "Yes" else 0

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✨ Analyze My Wellness", use_container_width=True, type="primary")

    # ── Result column ──────────────────────────────────────────────────────────
    with col_result:
        if submitted:
            x = np.array([[sleep, study, screen, social, diet, exercise_val]])
            try:
                x_scaled = scaler.transform(x)
                pred      = model.predict(x_scaled)
                label     = le.inverse_transform(pred)[0]

                # Gauge
                st.plotly_chart(
                    make_gauge(label),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                # Lifestyle gap cards
                gaps = []
                if sleep < 7:          gaps.append(("🛌 Sleep",    f"You sleep {sleep}h — aim for 7–9h."))
                if screen > 4:         gaps.append(("📱 Screen",   f"Screen time is {screen}h — try to stay under 4h."))
                if diet < 4:           gaps.append(("🍎 Diet",     f"Diet quality is {diet}/5 — aim for 4 or above."))
                if not exercise_val:   gaps.append(("🏃 Exercise", "Regular exercise significantly lowers stress."))
                if social < 2:         gaps.append(("🤝 Social",   "Low social activity increases isolation risk."))

                if gaps:
                    st.markdown(
                        "<div style='font-size:0.88rem; font-weight:600; color:#CBD5E1; margin-bottom:8px;'>🔍 Key Areas to Improve</div>",
                        unsafe_allow_html=True,
                    )
                    for icon, msg in gaps:
                        st.markdown(
                            f"""
                            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                                        border-radius:10px; padding:10px 14px; margin:6px 0;
                                        font-size:0.85rem; color:#94A3B8; line-height:1.5;'>
                                <b style='color:#CBD5E1;'>{icon}</b> &mdash; {msg}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("🌟 Excellent habits! Keep it up — your lifestyle looks very healthy.")

                st.info(
                    "💡 If you're feeling overwhelmed, consider reaching out to a trusted person "
                    "or a mental health professional."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown(
                """
                <div style='display:flex; flex-direction:column; align-items:center; justify-content:center;
                            height:340px; background:rgba(255,255,255,0.02);
                            border:1px dashed rgba(255,255,255,0.09);
                            border-radius:16px; text-align:center; padding:28px;'>
                    <div style='font-size:3rem; margin-bottom:14px; opacity:0.6;'>🎯</div>
                    <div style='font-size:0.92rem; color:#475569; max-width:260px; line-height:1.6;'>
                        Fill in your habits on the left and click
                        <b style="color:#6082B6;">Analyze My Wellness</b>
                        to see your personalised stress assessment.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Post-prediction: radar + comparison table ─────────────────────────────
    if submitted:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col_radar, col_table = st.columns([1, 1], gap="large")

        with col_radar:
            st.markdown("#### 🕸️ Habits Radar — You vs. Ideal")
            st.plotly_chart(
                make_radar(sleep, study, screen, social, diet, exercise_val),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_table:
            st.markdown("#### 📊 Inputs vs Recommended")
            st.dataframe(
                pd.DataFrame({
                    "Habit":       ["Sleep (hrs)", "Study/Work (hrs)", "Screen Time (hrs)",
                                    "Social Activity", "Diet Quality", "Exercise"],
                    "Your Value":  [sleep, study, screen, social, diet,
                                    "Yes" if exercise_val else "No"],
                    "Recommended": ["7–9 hrs", "≤8 hrs", "<4 hrs", "3–5", "≥4 / 5", "Yes"],
                    "Status":      [
                        "✅" if 7 <= sleep <= 9 else "⚠️",
                        "✅" if study <= 8       else "⚠️",
                        "✅" if screen < 4       else "⚠️",
                        "✅" if social >= 3      else "⚠️",
                        "✅" if diet >= 4        else "⚠️",
                        "✅" if exercise_val     else "⚠️",
                    ],
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("💡 Why These Habits Matter?"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🛌 Sleep**")
            st.caption("Essential for brain recovery. Consistent 7–9h sleep reduces cortisol and boosts cognitive function.")
        with c2:
            st.markdown("**📱 Screen Time**")
            st.caption("Excessive screens disrupt melatonin, degrade sleep quality, and can amplify anxiety.")
        with c3:
            st.markdown("**🏃 Exercise**")
            st.caption("Even a 20-min walk releases endorphins — your body's natural stress busters.")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("**🍎 Diet**")
            st.caption("Nutrition fuels your brain. Balanced meals stabilise mood, energy, and serotonin levels.")
        with c5:
            st.markdown("**🤝 Social**")
            st.caption("Meaningful connections reduce isolation and are the #1 predictor of long-term life satisfaction.")
        with c6:
            st.markdown("**📚 Study / Work**")
            st.caption("Over-studying without breaks causes burnout. The Pomodoro method helps balance focus and rest.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("<div class='section-label'>Data Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:1.9rem;'>Analytics & Feature Importance</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Understand which lifestyle factors your Random Forest model considers most influential.</div>",
        unsafe_allow_html=True,
    )

    feat_labels = ["Sleep", "Study/Work", "Screen Time", "Social Activity", "Diet Quality", "Exercise"]

    try:
        importances = model.feature_importances_

        col_bar, col_breakdown = st.columns([3, 2], gap="large")

        with col_bar:
            st.markdown("#### 📈 Feature Importances")
            st.plotly_chart(
                make_feature_bar(feat_labels, importances),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_breakdown:
            st.markdown("#### 📋 Importance Breakdown")
            total = importances.sum()
            for feat, imp in zip(feat_labels, importances):
                pct = imp / total * 100
                st.markdown(
                    f"""
                    <div style='margin-bottom:14px;'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                            <span style='color:#CBD5E1; font-size:0.88rem; font-weight:500;'>{feat}</span>
                            <span style='color:#6082B6; font-size:0.88rem; font-weight:600;'>{pct:.1f}%</span>
                        </div>
                        <div style='height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;'>
                            <div style='height:100%; width:{pct:.1f}%;
                                        background:linear-gradient(90deg,#6082B6,#3B5780);
                                        border-radius:3px;'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("#### 🍩 Importance Distribution")

        _, col_d, _ = st.columns([1, 2, 1])
        with col_d:
            st.plotly_chart(
                make_donut(feat_labels, importances),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Model info strip
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        top_feat = feat_labels[int(importances.argmax())]
        m1, m2, m3, m4 = st.columns(4)
        def metric_html(label, value, sub=""):
            return f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px; padding:18px; text-align:center;'>
                <div style='font-size:0.68rem; font-weight:700; letter-spacing:0.1em;
                            text-transform:uppercase; color:#6082B6; margin-bottom:6px;'>{label}</div>
                <div style='font-size:1.6rem; font-weight:700; color:#E2E8F0;'>{value}</div>
                {"<div style='font-size:0.75rem; color:#475569; margin-top:4px;'>" + sub + "</div>" if sub else ""}
            </div>
            """
        m1.markdown(metric_html("Model Type", "RF", "Random Forest"), unsafe_allow_html=True)
        m2.markdown(metric_html("Top Feature", top_feat, "highest importance"), unsafe_allow_html=True)
        m3.markdown(metric_html("Features", str(len(feat_labels)), "inputs"), unsafe_allow_html=True)
        m4.markdown(metric_html("Classes", "3", "Low · Med · High"), unsafe_allow_html=True)

    except AttributeError:
        st.warning("⚠️ Feature importances are not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TIPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Tips":
    st.markdown("<div class='section-label'>Wellness Resources</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:1.9rem;'>Tips for a Healthier Mind</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Small, consistent daily changes make the biggest long-term difference to mental wellness.</div>",
        unsafe_allow_html=True,
    )

    TIPS = [
        {
            "emoji": "📔", "title": "Gratitude Journaling",
            "tag": "Mindfulness", "hex": "6082B6",
            "body": "Write 3 things you're grateful for each morning. This shifts your brain's default mode from threat-detection to appreciation in as little as two weeks.",
        },
        {
            "emoji": "🌬️", "title": "Box Breathing",
            "tag": "Breathing", "hex": "7BA05B",
            "body": "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. This technique activates your parasympathetic nervous system and halts stress responses instantly.",
        },
        {
            "emoji": "📵", "title": "Digital Detox",
            "tag": "Screen Hygiene", "hex": "9F5B7B",
            "body": "Try a 1-hour phone-free window each morning. Reducing notification overload lowers cortisol, improves focus, and enhances the quality of your sleep.",
        },
        {
            "emoji": "🥗", "title": "Balanced Nutrition",
            "tag": "Nutrition", "hex": "5B9C9F",
            "body": "Omega-3 rich foods (walnuts, salmon) and leafy greens directly support serotonin production. Staying hydrated is equally vital for mood stability.",
        },
        {
            "emoji": "🏃", "title": "Move Your Body",
            "tag": "Exercise", "hex": "B79562",
            "body": "Even a 20-minute walk outdoors lowers cortisol, improves mood via endorphins, and enhances memory and learning retention.",
        },
        {
            "emoji": "🤝", "title": "Stay Connected",
            "tag": "Social", "hex": "5B7B9F",
            "body": "Schedule regular calls or meetups. Social bonds are consistently ranked the #1 predictor of life satisfaction in longitudinal studies.",
        },
    ]

    col1, col2, col3 = st.columns(3, gap="medium")
    cols = [col1, col2, col3]
    for i, tip in enumerate(TIPS):
        rgb = hex_rgb(tip["hex"])
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style='background:rgba(255,255,255,0.03);
                            border:1px solid rgba({rgb},0.22);
                            border-radius:16px; padding:22px; margin-bottom:16px;
                            transition: box-shadow 0.2s;'>
                    <div style='font-size:2rem; margin-bottom:10px;'>{tip["emoji"]}</div>
                    <div style='display:inline-block;
                                background:rgba({rgb},0.14);
                                color:#{tip["hex"]};
                                font-size:0.68rem; font-weight:700;
                                letter-spacing:0.08em; text-transform:uppercase;
                                padding:3px 10px; border-radius:20px; margin-bottom:10px;'>
                        {tip["tag"]}
                    </div>
                    <div style='font-size:0.98rem; font-weight:600;
                                color:#E2E8F0; margin-bottom:8px; line-height:1.3;'>
                        {tip["title"]}
                    </div>
                    <div style='font-size:0.83rem; color:#94A3B8; line-height:1.65;'>
                        {tip["body"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background:rgba(96,130,182,0.08); border:1px solid rgba(96,130,182,0.2);
                    border-radius:16px; padding:24px; text-align:center;'>
            <div style='font-size:1.3rem; font-weight:600; color:#E2E8F0; margin-bottom:8px;'>
                📞 Need Immediate Support?
            </div>
            <div style='color:#94A3B8; font-size:0.88rem; margin-bottom:14px;'>
                If you're in crisis, please reach out to a professional immediately.
            </div>
            <div style='display:flex; justify-content:center; gap:32px; flex-wrap:wrap;'>
                <div style='text-align:center;'>
                    <div style='font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
                                color:#475569; margin-bottom:4px;'>iCall</div>
                    <div style='color:#6082B6; font-weight:600; font-size:0.95rem;'>9152987821</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
                                color:#475569; margin-bottom:4px;'>Vandrevala Foundation</div>
                    <div style='color:#6082B6; font-weight:600; font-size:0.95rem;'>1860-2662-345</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase;
                                color:#475569; margin-bottom:4px;'>NIMHANS</div>
                    <div style='color:#6082B6; font-weight:600; font-size:0.95rem;'>080-46110007</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Chat":
    st.markdown("<div class='section-label'>Your AI Companion</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:1.9rem;'>Mental Wellness Chatbot</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Chat with our AI companion for advice, resources, or just to talk.</div>",
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY is not set. Please set it in your .env file to use the chatbot.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("How are you feeling today?"):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    client = genai.Client(api_key=api_key)
                    # Create prompt incorporating history
                    history_text = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
                    full_prompt = f"You are a helpful and empathetic mental wellness assistant. Respond to the user's latest message based on the conversation history.\\n\\nHistory:\\n{history_text}\\n\\nassistant: "
                    
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=full_prompt
                    )
                    
                    if response.text:
                        message_placeholder.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    else:
                        message_placeholder.markdown("*(No response received)*")
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗣️ Feedback":
    st.markdown("<div class='section-label'>Your Voice Matters</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title' style='font-size:1.9rem;'>Share Your Feedback</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Help shape the next version of MindPulse — every response makes a difference.</div>",
        unsafe_allow_html=True,
    )

    col_form, col_ratings = st.columns([3, 2], gap="large")

    with col_form:
        with st.form("feedback_form"):
            st.markdown("#### 📝 Your Experience")

            rating = st.select_slider(
                "⭐ Overall Rating",
                options=["1 — Poor", "2 — Fair", "3 — Good", "4 — Great", "5 — Excellent"],
                value="4 — Great",
            )

            accuracy = st.radio(
                "🎯 Did the stress prediction feel accurate?",
                ["Yes, very accurate", "Somewhat accurate", "Not accurate", "Unsure"],
                horizontal=True,
            )

            feature_req = st.multiselect(
                "✨ Features you'd like to see next",
                [
                    "Progress tracking over time",
                    "Daily mood check-in",
                    "Sleep quality tracker",
                    "Guided meditation",
                    "Peer support forum",
                    "Therapist directory",
                    "AI chat support",
                ],
            )

            suggestions = st.text_area(
                "💬 Suggestions or comments",
                placeholder="Share your thoughts, ideas, or any bugs you noticed…",
                height=120,
            )

            sub_fb = st.form_submit_button("📤 Submit Feedback", use_container_width=True)

        if sub_fb:
            st.success("🎉 Thank you for your feedback! It directly shapes the next version of MindPulse.")

    with col_ratings:
        st.markdown("#### 📊 Community Ratings")
        RATINGS = {
            "Ease of Use":  87,
            "Accuracy":     74,
            "Design":       91,
            "Helpfulness":  79,
            "Overall":      83,
        }
        for label, pct in RATINGS.items():
            color = "#22C55E" if pct >= 80 else "#F59E0B" if pct >= 65 else "#EF4444"
            st.markdown(
                f"""
                <div style='margin-bottom:16px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:7px;'>
                        <span style='color:#CBD5E1; font-size:0.88rem; font-weight:500;'>{label}</span>
                        <span style='color:{color}; font-size:0.88rem; font-weight:700;'>{pct}%</span>
                    </div>
                    <div style='height:7px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden;'>
                        <div style='height:100%; width:{pct}%; background:{color};
                                    border-radius:4px; opacity:0.75;'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='background:rgba(123,160,91,0.09); border:1px solid rgba(123,160,91,0.22);
                        border-radius:14px; padding:18px;'>
                <div style='font-size:0.88rem; color:#94A3B8; line-height:1.65;'>
                    <b style='color:#7BA05B;'>🌱 Growing Together</b><br/>
                    MindPulse is a student project built for awareness.
                    Your feedback directly shapes the next version.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='margin-top:48px; padding-top:18px;
                border-top:1px solid rgba(255,255,255,0.06); text-align:center;'>
        <span style='font-size:0.75rem; color:#334155;'>
            MindPulse &nbsp;·&nbsp; Mental Health Awareness &nbsp;·&nbsp; © 2025 Dev Saxena
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
