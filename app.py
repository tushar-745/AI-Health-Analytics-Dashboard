import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import base64
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="HealthAI Dashboard",
    page_icon="assets/overview.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ICON HELPERS ──────────────────────────────────────────────
@st.cache_data
def load_icon_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

ICON_OVERVIEW   = load_icon_b64("assets/dashboard.png")   # Overview page
ICON_HEART      = load_icon_b64("assets/business.png")    # Heart Disease Predictor
ICON_DIABETES   = load_icon_b64("assets/healthcare.png")  # Diabetes Predictor
ICON_PATIENTS   = load_icon_b64("assets/overview.png")    # Patient Analytics
ICON_PERFORMANCE= load_icon_b64("assets/forecast.png")    # Model Performance
ICON_LOGO       = load_icon_b64("assets/healthcare.png")  # Sidebar brand logo

def header_icon_img(b64_str, size=40):
    # icons are black line-art on transparent bg; invert to white so they
    # show up against the dark navy header-strip background
    return f'<img src="data:image/png;base64,{b64_str}" style="width:{size}px;height:{size}px;filter:invert(1) brightness(1.6);flex-shrink:0;">'

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Source+Serif+Pro:wght@600;700&display=swap');

:root {
    --brand-navy:   #0B1526;
    --brand-navy-2: #101E36;
    --brand-blue:   #2454E8;
    --brand-blue-d: #1B3FBE;
    --ink:          #101828;
    --ink-soft:     #475467;
    --ink-faint:    #667085;
    --green:        #0F9D58;
    --red:          #D93025;
    --amber:        #C77700;
    --purple:       #6941C6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
/* CHANGED: solid, slightly less saturated navy for reliable contrast */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--brand-navy) 0%, var(--brand-navy-2) 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #DCE3F0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    padding: 6px 0;
}

/* CHANGED: This is the actual fix — the sidebar numbers are wrapped in
   backticks (`{value}`), which Streamlit renders as inline <code>.
   By default that's a light-gray box with barely-darker gray text,
   which on a dark sidebar looks washed out and hard to read.
   Force it into a solid white pill with dark navy text — the highest
   contrast pairing against the dark sidebar background. */
section[data-testid="stSidebar"] code {
    background: #FFFFFF !important;
    color: var(--brand-navy) !important;
    font-weight: 800 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 20px !important;
    padding: 3px 12px !important;
    font-size: 0.85rem !important;
}

/* Main background */
.main { background: #F4F6FA; }

/* Cards */
/* CHANGED: bolder value text, muted professional accent colors, subtle border */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(15,40,80,0.08);
    border: 1px solid #E4E9F2;
    border-left: 4px solid var(--brand-blue);
    margin-bottom: 8px;
}
.kpi-card.green  { border-left-color: var(--green); }
.kpi-card.red    { border-left-color: var(--red); }
.kpi-card.orange { border-left-color: var(--amber); }
.kpi-card.purple { border-left-color: var(--purple); }

.kpi-value { font-size: 2rem; font-weight: 800; color: var(--ink); line-height: 1; }
.kpi-label { font-size: 0.78rem; font-weight: 700; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
.kpi-delta { font-size: 0.82rem; font-weight: 600; color: var(--red); margin-top: 6px; }

/* Section titles */
.section-title {
    font-family: 'Source Serif Pro', serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: var(--ink);
    margin-bottom: 4px;
}
.section-sub { font-size: 0.88rem; color: var(--ink-faint); margin-bottom: 20px; }

/* Prediction result box */
.pred-box {
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    margin-top: 12px;
}
.pred-high { background: #FDECEC; border: 2px solid var(--red); }
.pred-low  { background: #E9F8EF; border: 2px solid var(--green); }
.pred-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
.pred-pct   { font-size: 2.5rem; font-weight: 800; }
.pred-high .pred-title, .pred-high .pred-pct { color: #A31C13; }
.pred-low  .pred-title, .pred-low  .pred-pct { color: #0B7A3F; }

/* Model badge */
.model-badge {
    display: inline-block;
    background: #E9F0FE;
    color: var(--brand-blue-d);
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* Top header strip */
.header-strip {
    background: linear-gradient(90deg, var(--brand-navy) 0%, #1a3a6b 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-title {
    font-family: 'Source Serif Pro', serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: white;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-sub { color: #9FB6E0; font-size: 0.9rem; margin-top: 4px; }

/* Sidebar brand logo */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.sidebar-logo img {
    width: 32px;
    height: 32px;
    filter: invert(1) brightness(1.6);
    flex-shrink: 0;
}
.sidebar-logo span {
    font-size: 1.25rem;
    font-weight: 800;
    color: white;
}

/* Sidebar navigation */
.nav-heading {
    font-size: 0.8rem;
    font-weight: 700;
    color: #9FB6E0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.nav-list { display: flex; flex-direction: column; gap: 3px; margin-bottom: 4px; }
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 8px;
    text-decoration: none !important;
    color: #DCE3F0 !important;
    transition: background 0.15s ease;
}
.nav-item:hover { background: rgba(255,255,255,0.07); }
.nav-item.active { background: rgba(36,84,232,0.22); }
.nav-dot {
    width: 9px;
    height: 9px;
    min-width: 9px;
    border-radius: 50%;
    border: 1.5px solid #6b7f9e;
    flex-shrink: 0;
}
.nav-dot.active { background: #E5484D; border-color: #E5484D; }
.nav-icon {
    width: 18px;
    height: 18px;
    filter: invert(1) brightness(1.6);
    flex-shrink: 0;
}
.nav-label { font-size: 0.92rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA & MODELS ────────────────────────────────────────
@st.cache_data
def load_data():
    df_diabetes  = pd.read_csv("data/diabetes.csv")
    df_heart     = pd.read_csv("data/heart.csv")
    df_patients  = pd.read_csv("data/patients.csv", parse_dates=["AdmissionDate"])
    return df_diabetes, df_heart, df_patients

@st.cache_resource
def load_models():
    with open("models/diabetes_model.pkl",   "rb") as f: dm = pickle.load(f)
    with open("models/diabetes_scaler.pkl",  "rb") as f: ds = pickle.load(f)
    with open("models/heart_model.pkl",      "rb") as f: hm = pickle.load(f)
    with open("models/heart_scaler.pkl",     "rb") as f: hs = pickle.load(f)
    with open("models/diabetes_features.pkl","rb") as f: df_feat = pickle.load(f)
    with open("models/heart_features.pkl",   "rb") as f: hf_feat = pickle.load(f)
    with open("models/metrics.pkl",          "rb") as f: metrics = pickle.load(f)
    return dm, ds, hm, hs, df_feat, hf_feat, metrics

df_diabetes, df_heart, df_patients = load_data()
d_model, d_scaler, h_model, h_scaler, d_feat, h_feat, metrics = load_models()

# ── SIDEBAR ───────────────────────────────────────────────────
NAV_ITEMS = [
    ("Overview",                 "Overview", ICON_OVERVIEW),
    ("Heart Disease Predictor",  "Heart",    ICON_HEART),
    ("Diabetes Predictor",       "Diabetes", ICON_DIABETES),
    ("Patient Analytics",        "Patient",  ICON_PATIENTS),
    ("Model Performance",        "Model",    ICON_PERFORMANCE),
]
_valid_keys = [k for _, k, _ in NAV_ITEMS]
page = st.query_params.get("nav", "Overview")
if page not in _valid_keys:
    page = "Overview"

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{ICON_LOGO}">
        <span>HealthAI</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="nav-heading">Navigation</div>', unsafe_allow_html=True)
    nav_html = '<div class="nav-list">'
    for label, key, icon_b64 in NAV_ITEMS:
        active = (key == page)
        item_cls = "nav-item active" if active else "nav-item"
        dot_cls  = "nav-dot active" if active else "nav-dot"
        nav_html += f'''<a href="?nav={key}" target="_self" class="{item_cls}">
            <span class="{dot_cls}"></span>
            <img class="nav-icon" src="data:image/png;base64,{icon_b64}">
            <span class="nav-label">{label}</span>
        </a>'''
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Data Summary**")
    st.markdown(f"- Patients: `{len(df_patients):,}`")
    st.markdown(f"- Diabetes records: `{len(df_diabetes):,}`")
    st.markdown(f"- Heart records: `{len(df_heart):,}`")
    st.markdown("---")
    st.markdown("<span style='font-size:0.75rem;color:#7C8CB0'>HealthAI v1.0 · ML-Powered</span>", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────
CHART_CONFIG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#3d5166"),
    margin=dict(t=40, b=20, l=10, r=10),
)

def card(value, label, color="blue", delta=""):
    return f"""
    <div class="kpi-card {color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {"<div class='kpi-delta'>"+delta+"</div>" if delta else ""}
    </div>"""

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown(f"""
    <div class="header-strip">
        <div>
            <div class="header-title">{header_icon_img(ICON_OVERVIEW)} HealthAI Dashboard</div>
            <div class="header-sub">Real-time analytics · Disease prediction · Patient insights</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    total_p   = len(df_patients)
    critical  = (df_patients["Status"] == "Critical").sum()
    avg_los   = df_patients["LengthOfStay"].mean()
    avg_chg   = df_patients["TotalCharges"].mean()
    diab_rate = df_diabetes["Outcome"].mean() * 100
    heart_rate= df_heart["target"].mean() * 100

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(card(f"{total_p:,}",   "Total Patients",      "blue"),   unsafe_allow_html=True)
    c2.markdown(card(f"{critical}",    "Critical Cases",      "red",   "⚠ Needs attention"), unsafe_allow_html=True)
    c3.markdown(card(f"{avg_los:.1f}d","Avg Length of Stay",  "orange"),unsafe_allow_html=True)
    c4.markdown(card(f"₹{avg_chg:,.0f}","Avg Treatment Cost", "purple"),unsafe_allow_html=True)
    c5.markdown(card(f"{diab_rate:.1f}%","Diabetes Prevalence","green"), unsafe_allow_html=True)
    c6.markdown(card(f"{heart_rate:.1f}%","Heart Risk Rate",  "blue"),  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Department Load</div>', unsafe_allow_html=True)
        dept_counts = df_patients["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig = px.bar(dept_counts, x="Count", y="Department", orientation="h",
                     color="Count", color_continuous_scale=["#8bafd4","#2454E8"],
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_CONFIG, showlegend=False, coloraxis_showscale=False, height=320)
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Patient Status Breakdown</div>', unsafe_allow_html=True)
        status_counts = df_patients["Status"].value_counts()
        colors = {"Admitted":"#2454E8","Discharged":"#0F9D58","Critical":"#D93025","Stable":"#C77700"}
        fig2 = px.pie(values=status_counts.values, names=status_counts.index,
                      color=status_counts.index,
                      color_discrete_map=colors, hole=0.55)
        fig2.update_traces(textinfo="label+percent", textfont_size=12)
        fig2.update_layout(**CHART_CONFIG, height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Age Distribution by Gender</div>', unsafe_allow_html=True)
        fig3 = px.histogram(df_patients, x="Age", color="Gender",
                            barmode="overlay", nbins=30,
                            color_discrete_map={"Male":"#2454E8","Female":"#B5459A"},
                            opacity=0.75)
        fig3.update_layout(**CHART_CONFIG, height=300, bargap=0.05)
        fig3.update_xaxes(showgrid=False)
        fig3.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Monthly Admissions Trend</div>', unsafe_allow_html=True)
        df_patients["YearMonth"] = df_patients["AdmissionDate"].dt.to_period("M").astype(str)
        monthly = df_patients.groupby("YearMonth").size().reset_index(name="Admissions")
        monthly = monthly.tail(24)
        fig4 = px.line(monthly, x="YearMonth", y="Admissions",
                       markers=True, color_discrete_sequence=["#2454E8"])
        fig4.update_traces(line_width=2.5, marker_size=5)
        fig4.update_layout(**CHART_CONFIG, height=300)
        fig4.update_xaxes(showgrid=False, tickangle=45, nticks=8)
        fig4.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — HEART DISEASE PREDICTOR
# ══════════════════════════════════════════════════════════════
elif "Heart" in page:
    st.markdown(f"""
    <div class="header-strip">
        <div>
            <div class="header-title">{header_icon_img(ICON_HEART)} Heart Disease Risk Predictor</div>
            <div class="header-sub">Gradient Boosting model · AUC 0.91 · Enter patient vitals below</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-title">Patient Vitals</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Fill in the clinical measurements</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        age      = r1c1.slider("Age", 20, 80, 52)
        sex      = r1c2.selectbox("Sex", ["Male (1)", "Female (0)"])
        sex_val  = 1 if "Male" in sex else 0

        r2c1, r2c2 = st.columns(2)
        cp       = r2c1.selectbox("Chest Pain Type", [
                        "Typical Angina (0)", "Atypical Angina (1)",
                        "Non-anginal (2)", "Asymptomatic (3)"])
        cp_val   = int(cp.split("(")[1].rstrip(")"))
        trestbps = r2c2.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)

        r3c1, r3c2 = st.columns(2)
        chol     = r3c1.slider("Cholesterol (mg/dl)", 150, 400, 240)
        thalach  = r3c2.slider("Max Heart Rate Achieved", 70, 210, 152)

        r4c1, r4c2 = st.columns(2)
        exang    = r4c1.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        exang_v  = int(exang.split("(")[1].rstrip(")"))
        oldpeak  = r4c2.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

        r5c1, r5c2, r5c3 = st.columns(3)
        fbs      = r5c1.selectbox("Fasting BS > 120", ["No (0)", "Yes (1)"])
        fbs_v    = int(fbs.split("(")[1].rstrip(")"))
        restecg  = r5c2.selectbox("Resting ECG", ["Normal (0)", "ST-T (1)", "LVH (2)"])
        restecg_v= int(restecg.split("(")[1].rstrip(")"))
        slope    = r5c3.selectbox("Slope of ST", ["Up (0)", "Flat (1)", "Down (2)"])
        slope_v  = int(slope.split("(")[1].rstrip(")"))

        r6c1, r6c2 = st.columns(2)
        ca       = r6c1.slider("Major Vessels (ca)", 0, 4, 0)
        thal     = r6c2.selectbox("Thalassemia", ["Normal (1)", "Fixed (2)", "Reversible (3)"])
        thal_v   = int(thal.split("(")[1].rstrip(")"))

        predict_btn = st.button("🔍 Predict Heart Risk", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            inp = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_v, restecg_v,
                             thalach, exang_v, oldpeak, slope_v, ca, thal_v]])
            inp_s = h_scaler.transform(inp)
            prob  = h_model.predict_proba(inp_s)[0][1]
            pct   = int(prob * 100)
            risk  = "HIGH" if prob >= 0.5 else "LOW"
            css   = "pred-high" if risk == "HIGH" else "pred-low"
            icon  = "⚠️" if risk == "HIGH" else "✅"

            st.markdown(f"""
            <div class="pred-box {css}">
                <div class="pred-title">{icon} {risk} RISK</div>
                <div class="pred-pct">{pct}%</div>
                <div style="font-size:0.85rem;margin-top:8px;opacity:0.75">probability of heart disease</div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix":"%","font":{"size":32}},
                gauge={
                    "axis": {"range":[0,100], "tickwidth":1},
                    "bar":  {"color": "#D93025" if risk=="HIGH" else "#0F9D58"},
                    "steps":[
                        {"range":[0,33],   "color":"#EAF7EF"},
                        {"range":[33,66],  "color":"#FDF3E3"},
                        {"range":[66,100], "color":"#FBEAE9"},
                    ],
                    "threshold":{"line":{"color":"#101828","width":3},"thickness":0.75,"value":50}
                }
            ))
            fig_g.update_layout(height=260, **CHART_CONFIG)
            st.plotly_chart(fig_g, use_container_width=True)

            # Feature importance bar
            importances = h_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": h_feat, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(8)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale=["#8bafd4","#2454E8"])
            fig_fi.update_layout(**CHART_CONFIG, height=260, showlegend=False,
                                 coloraxis_showscale=False,
                                 title="Top Contributing Factors")
            fig_fi.update_xaxes(showgrid=False, visible=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        else:
            st.info("👈 Fill in the patient vitals and click **Predict Heart Risk**")

            # Distribution context
            st.markdown("**Heart Disease Prevalence in Dataset**")
            hd = df_heart["target"].value_counts().reset_index()
            hd.columns = ["Outcome", "Count"]
            hd["Outcome"] = hd["Outcome"].map({0:"No Disease", 1:"Has Disease"})
            fig_hd = px.pie(hd, values="Count", names="Outcome",
                            color_discrete_sequence=["#0F9D58","#D93025"], hole=0.5)
            fig_hd.update_layout(**CHART_CONFIG, height=300, showlegend=True)
            st.plotly_chart(fig_hd, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — DIABETES PREDICTOR
# ══════════════════════════════════════════════════════════════
elif "Diabetes" in page:
    st.markdown(f"""
    <div class="header-strip">
        <div>
            <div class="header-title">{header_icon_img(ICON_DIABETES)} Diabetes Risk Predictor</div>
            <div class="header-sub">Random Forest model · AUC 0.92 · Enter patient measurements below</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-title">Patient Measurements</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Fill in the clinical details</div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        pregnancies = r1c1.slider("Pregnancies", 0, 17, 1)
        glucose     = r1c2.slider("Glucose Level (mg/dL)", 0, 200, 110)

        r2c1, r2c2 = st.columns(2)
        bp          = r2c1.slider("Blood Pressure (mmHg)", 0, 130, 72)
        skin_thick  = r2c2.slider("Skin Thickness (mm)", 0, 60, 20)

        r3c1, r3c2 = st.columns(2)
        insulin     = r3c1.slider("Insulin (μU/mL)", 0, 300, 80)
        bmi         = r3c2.slider("BMI", 10.0, 55.0, 25.0, step=0.1)

        r4c1, r4c2 = st.columns(2)
        dpf         = r4c1.slider("Diabetes Pedigree Function", 0.08, 2.5, 0.47, step=0.01)
        age         = r4c2.slider("Age", 20, 81, 33)

        predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            inp   = np.array([[pregnancies, glucose, bp, skin_thick, insulin, bmi, dpf, age]])
            inp_s = d_scaler.transform(inp)
            prob  = d_model.predict_proba(inp_s)[0][1]
            pct   = int(prob * 100)
            risk  = "HIGH" if prob >= 0.5 else "LOW"
            css   = "pred-high" if risk == "HIGH" else "pred-low"
            icon  = "⚠️" if risk == "HIGH" else "✅"

            st.markdown(f"""
            <div class="pred-box {css}">
                <div class="pred-title">{icon} {risk} RISK</div>
                <div class="pred-pct">{pct}%</div>
                <div style="font-size:0.85rem;margin-top:8px;opacity:0.75">probability of diabetes</div>
            </div>""", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix":"%","font":{"size":32}},
                gauge={
                    "axis": {"range":[0,100],"tickwidth":1},
                    "bar":  {"color":"#D93025" if risk=="HIGH" else "#0F9D58"},
                    "steps":[
                        {"range":[0,33],   "color":"#EAF7EF"},
                        {"range":[33,66],  "color":"#FDF3E3"},
                        {"range":[66,100], "color":"#FBEAE9"},
                    ],
                    "threshold":{"line":{"color":"#101828","width":3},"thickness":0.75,"value":50}
                }
            ))
            fig_g.update_layout(height=260, **CHART_CONFIG)
            st.plotly_chart(fig_g, use_container_width=True)

            importances = d_model.feature_importances_
            fi_df = pd.DataFrame({"Feature": d_feat, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale=["#b4d4f7","#B5459A"])
            fig_fi.update_layout(**CHART_CONFIG, height=280, showlegend=False,
                                 coloraxis_showscale=False,
                                 title="Top Contributing Factors")
            fig_fi.update_xaxes(showgrid=False, visible=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        else:
            st.info("👈 Fill in the patient measurements and click **Predict Diabetes Risk**")

            st.markdown("**Glucose vs BMI · Diabetes Outcome**")
            fig_sc = px.scatter(df_diabetes, x="Glucose", y="BMI",
                                color=df_diabetes["Outcome"].map({0:"No Diabetes", 1:"Diabetes"}),
                                color_discrete_map={"No Diabetes":"#0F9D58","Diabetes":"#D93025"},
                                opacity=0.6, size_max=6)
            fig_sc.update_layout(**CHART_CONFIG, height=340, legend_title="")
            fig_sc.update_xaxes(showgrid=True, gridcolor="#eef2f7")
            fig_sc.update_yaxes(showgrid=True, gridcolor="#eef2f7")
            st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — PATIENT ANALYTICS
# ══════════════════════════════════════════════════════════════
elif "Patient" in page:
    st.markdown(f"""
    <div class="header-strip">
        <div>
            <div class="header-title">{header_icon_img(ICON_PATIENTS)} Patient Analytics</div>
            <div class="header-sub">Hospital operations · Costs · Satisfaction · Department insights</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    dept_filter   = fcol1.multiselect("Department", df_patients["Department"].unique(),
                                       default=list(df_patients["Department"].unique()))
    status_filter = fcol2.multiselect("Status", df_patients["Status"].unique(),
                                       default=list(df_patients["Status"].unique()))
    gender_filter = fcol3.multiselect("Gender", df_patients["Gender"].unique(),
                                       default=list(df_patients["Gender"].unique()))

    fdf = df_patients[
        df_patients["Department"].isin(dept_filter) &
        df_patients["Status"].isin(status_filter) &
        df_patients["Gender"].isin(gender_filter)
    ]

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(card(f"{len(fdf):,}", "Filtered Patients", "blue"), unsafe_allow_html=True)
    k2.markdown(card(f"₹{fdf['TotalCharges'].mean():,.0f}", "Avg Charges", "purple"), unsafe_allow_html=True)
    k3.markdown(card(f"{fdf['LengthOfStay'].mean():.1f}d", "Avg Stay", "orange"), unsafe_allow_html=True)
    k4.markdown(card(f"{fdf['Satisfaction'].mean():.1f}/5", "Avg Satisfaction", "green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)

    with row1c1:
        st.markdown('<div class="section-title">Avg Treatment Cost by Department</div>', unsafe_allow_html=True)
        dept_cost = fdf.groupby("Department")["TotalCharges"].mean().reset_index().sort_values("TotalCharges")
        fig5 = px.bar(dept_cost, x="TotalCharges", y="Department", orientation="h",
                      color="TotalCharges", color_continuous_scale=["#8bafd4","#6941C6"],
                      text=dept_cost["TotalCharges"].apply(lambda x: f"₹{x:,.0f}"))
        fig5.update_traces(textposition="outside")
        fig5.update_layout(**CHART_CONFIG, height=320, showlegend=False, coloraxis_showscale=False)
        fig5.update_xaxes(showgrid=False, visible=False)
        st.plotly_chart(fig5, use_container_width=True)

    with row1c2:
        st.markdown('<div class="section-title">Length of Stay Distribution</div>', unsafe_allow_html=True)
        fig6 = px.violin(fdf, x="Department", y="LengthOfStay", color="Department",
                         box=True, points=False,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig6.update_layout(**CHART_CONFIG, height=320, showlegend=False)
        fig6.update_xaxes(tickangle=20)
        fig6.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig6, use_container_width=True)

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        st.markdown('<div class="section-title">Patient Satisfaction by Department</div>', unsafe_allow_html=True)
        sat = fdf.groupby("Department")["Satisfaction"].mean().reset_index()
        sat["Color"] = sat["Satisfaction"].apply(lambda x: "#0F9D58" if x>=4 else ("#C77700" if x>=3 else "#D93025"))
        fig7 = px.bar(sat, x="Department", y="Satisfaction",
                      color="Department", text=sat["Satisfaction"].round(2),
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig7.update_traces(textposition="outside")
        fig7.update_layout(**CHART_CONFIG, height=300, showlegend=False)
        fig7.update_yaxes(range=[0,5.5], showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig7, use_container_width=True)

    with row2c2:
        st.markdown('<div class="section-title">BMI vs Cholesterol (by Gender)</div>', unsafe_allow_html=True)
        fig8 = px.scatter(fdf.sample(min(400,len(fdf))), x="BMI", y="Cholesterol",
                          color="Gender", opacity=0.65,
                          color_discrete_map={"Male":"#2454E8","Female":"#B5459A"})
        fig8.update_layout(**CHART_CONFIG, height=300)
        fig8.update_xaxes(showgrid=True, gridcolor="#eef2f7")
        fig8.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig8, use_container_width=True)

    # Patient table
    st.markdown('<div class="section-title">Patient Records</div>', unsafe_allow_html=True)
    show_cols = ["PatientID","Gender","Age","Department","Status","LengthOfStay","TotalCharges","Satisfaction"]
    st.dataframe(
        fdf[show_cols].sort_values("TotalCharges", ascending=False).head(50).reset_index(drop=True),
        use_container_width=True, height=320
    )

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown(f"""
    <div class="header-strip">
        <div>
            <div class="header-title">{header_icon_img(ICON_PERFORMANCE)} Model Performance</div>
            <div class="header-sub">Accuracy · AUC · Feature importance · Data distributions</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Metrics cards
    m1,m2,m3,m4 = st.columns(4)
    m1.markdown(card(f"{metrics['diabetes']['acc']*100:.1f}%", "Diabetes Accuracy", "blue"),  unsafe_allow_html=True)
    m2.markdown(card(f"{metrics['diabetes']['auc']:.3f}",      "Diabetes AUC",      "green"), unsafe_allow_html=True)
    m3.markdown(card(f"{metrics['heart']['acc']*100:.1f}%",    "Heart Accuracy",    "orange"),unsafe_allow_html=True)
    m4.markdown(card(f"{metrics['heart']['auc']:.3f}",         "Heart AUC",         "purple"),unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Diabetes — Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({"Feature": d_feat,
                           "Importance": d_model.feature_importances_}).sort_values("Importance")
        fig_d = px.bar(fi, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale=["#b4d4f7","#B5459A"])
        fig_d.update_layout(**CHART_CONFIG, height=320, coloraxis_showscale=False)
        fig_d.update_xaxes(showgrid=False)
        st.plotly_chart(fig_d, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Heart Disease — Feature Importance</div>', unsafe_allow_html=True)
        fi2 = pd.DataFrame({"Feature": h_feat,
                            "Importance": h_model.feature_importances_}).sort_values("Importance")
        fig_h = px.bar(fi2, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale=["#8bafd4","#2454E8"])
        fig_h.update_layout(**CHART_CONFIG, height=320, coloraxis_showscale=False)
        fig_h.update_xaxes(showgrid=False)
        st.plotly_chart(fig_h, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Glucose Distribution · Diabetes</div>', unsafe_allow_html=True)
        fig_glu = px.histogram(df_diabetes, x="Glucose",
                               color=df_diabetes["Outcome"].map({0:"No Diabetes",1:"Diabetes"}),
                               barmode="overlay", nbins=40, opacity=0.75,
                               color_discrete_map={"No Diabetes":"#0F9D58","Diabetes":"#D93025"})
        fig_glu.update_layout(**CHART_CONFIG, height=280, legend_title="")
        fig_glu.update_xaxes(showgrid=False)
        fig_glu.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig_glu, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Age Distribution · Heart Disease</div>', unsafe_allow_html=True)
        fig_age = px.histogram(df_heart, x="age",
                               color=df_heart["target"].map({0:"No Disease",1:"Has Disease"}),
                               barmode="overlay", nbins=40, opacity=0.75,
                               color_discrete_map={"No Disease":"#0F9D58","Has Disease":"#D93025"})
        fig_age.update_layout(**CHART_CONFIG, height=280, legend_title="")
        fig_age.update_xaxes(showgrid=False)
        fig_age.update_yaxes(showgrid=True, gridcolor="#eef2f7")
        st.plotly_chart(fig_age, use_container_width=True)

    # Model info table
    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Disease": ["Diabetes", "Heart Disease"],
        "Algorithm": ["Random Forest (100 trees)", "Gradient Boosting (100 trees)"],
        "Features": [len(d_feat), len(h_feat)],
        "Training Samples": [800, 800],
        "Test Accuracy": [f"{metrics['diabetes']['acc']*100:.1f}%", f"{metrics['heart']['acc']*100:.1f}%"],
        "AUC Score": [f"{metrics['diabetes']['auc']:.3f}", f"{metrics['heart']['auc']:.3f}"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)