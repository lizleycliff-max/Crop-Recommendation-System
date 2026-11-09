"""
Crop Recommendation - Tamil Nadu
Explainable AI for crop recommendation - sidebar-navigated Streamlit app.

Pages: Home | Crop Recommendation | Explainable AI | Visualizations |
       What-If Analysis | Model Performance | Dataset Explorer

Model, preprocessing and explainer are used exactly as trained (src.explain / src.preprocess).
Every number shown comes from the model output, the saved metric files, or the dataset.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explain import CropExplainer
from src.preprocess import NUMERICAL_COLS

APP_NAME = "Crop Recommendation"
APP_TAGLINE = "Know the right crop for your field, and why"

st.set_page_config(page_title=f"{APP_NAME} | Tamil Nadu", page_icon="🌱",
                   layout="wide", initial_sidebar_state="expanded")

GREEN_D, GREEN, GREEN_M, GREEN_L = "#24352B", "#3F7D58", "#86B896", "#DCEADF"
EARTH, EARTH_L, GREY, RED = "#C79A45", "#F8F3E6", "#A8ADA3", "#B5523F"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stApp {{ font-family:'Inter',-apple-system,sans-serif; font-size:16px; }}
    footer, #MainMenu {{ visibility:hidden; }}
    .block-container {{ padding-top:4.5rem; padding-bottom:3rem; max-width:1150px; }}
    h1, h2, h3 {{ color:{GREEN_D}; font-weight:600; }}

    /* ---- sidebar: light, quiet */
    [data-testid="stSidebar"] {{ background:#F7F8F4; border-right:1px solid #E7E9E1; }}
    [data-testid="stSidebar"] * {{ color:#3C4A41; }}
    [data-testid="stSidebar"] hr {{ border-color:#E7E9E1; }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.1rem; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding:.5rem .8rem; border-radius:8px; width:100%; cursor:pointer;
        border-left:3px solid transparent; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:#EEF1E9; }}
    /* hide the radio dot (several selectors to survive Streamlit version changes) */
    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child,
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
    [data-testid="stSidebar"] div[role="radiogroup"] label span[data-baseweb="radio"],
    [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {{
        display:none !important; width:0 !important; height:0 !important; margin:0 !important; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:.98rem; font-weight:400; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background:#E3EDE3; border-left:3px solid {GREEN}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
        color:{GREEN_D}; font-weight:600; }}
    .sb-brand {{ font-size:1.25rem; font-weight:600; color:{GREEN_D} !important; line-height:1.2; }}
    .sb-sub {{ font-size:.8rem; color:#7A857C !important; margin-bottom:.3rem; }}
    .sb-now {{ background:#fff; border:1px solid #E7E9E1; border-radius:8px; padding:.65rem .8rem;
               font-size:.85rem; color:#6B766D; }}
    .sb-now b {{ color:{GREEN_D} !important; font-size:1rem; font-weight:600; }}

    /* ---- accent colours (backup to the theme setting) */
    div[data-baseweb="tab-highlight"] {{ background-color:{GREEN} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] p {{ color:{GREEN} !important; }}
    div[data-testid="stSlider"] [role="slider"] {{ background-color:{GREEN} !important; }}
    div[data-testid="stSliderThumbValue"] p, div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {{
        color:{GREEN} !important; font-size:.85rem; }}

    /* ---- page header */
    .eyebrow {{ font-size:.75rem; letter-spacing:.14em; color:{GREEN}; font-weight:600; margin-bottom:.15rem; }}
    .page-title {{ font-size:1.9rem; font-weight:600; color:{GREEN_D}; margin:0; line-height:1.25; }}
    .page-sub {{ font-size:1rem; color:#6b7280; margin:.3rem 0 1.3rem; }}
    .sec-title {{ font-size:1.15rem; font-weight:600; color:{GREEN_D}; margin:1.2rem 0 .1rem; }}
    .sec-sub {{ font-size:.9rem; color:#6b7280; margin-bottom:.6rem; }}

    /* ---- hero: soft, light */
    .hero {{ background:#F4F7F1; border:1px solid #E1EBDD; border-radius:14px;
             padding:2.4rem 2.6rem 2.1rem; margin-bottom:1.4rem; }}
    .hero-eyebrow {{ font-size:.8rem; letter-spacing:.14em; text-transform:uppercase;
                     color:{GREEN}; font-weight:600; }}
    .hero-title {{ font-size:2.8rem; font-weight:600; line-height:1.1; margin:.4rem 0 .2rem; color:{GREEN_D}; }}
    .hero-ta {{ font-size:1.6rem; color:#7aa58b; font-weight:400; margin-left:.4rem; }}
    .hero-tag {{ font-size:1.2rem; color:{GREEN}; margin-bottom:.8rem; font-weight:500; }}
    .hero-text {{ font-size:1.02rem; color:#56625A; max-width:680px; line-height:1.6; }}
    .chip {{ display:inline-block; background:#fff; color:{GREEN}; border:1px solid #D7E3D3;
             border-radius:999px; padding:.2rem .8rem; font-size:.82rem; margin:1rem .4rem 0 0; }}

    /* ---- cards */
    .stat {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:1rem 1.2rem; }}
    .stat-v {{ font-size:1.7rem; font-weight:600; color:{GREEN_D}; line-height:1.1; }}
    .stat-l {{ font-size:.85rem; color:#6b7280; margin-top:.25rem; }}
    .feat {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:1.1rem 1.25rem .6rem;
             min-height:122px; }}
    .feat-t {{ font-size:1.05rem; font-weight:600; color:{GREEN_D}; }}
    .feat-d {{ font-size:.92rem; color:#6b7280; margin-top:.25rem; line-height:1.45; }}
    .crop-card {{ background:#F9FBF7; border:1px solid #E1EBDD; border-left:4px solid {GREEN_M};
                  border-radius:12px; padding:1.3rem 1.5rem; }}
    .crop-eyebrow {{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase;
                     color:{GREEN}; font-weight:600; }}
    .crop-name {{ font-size:2.1rem; font-weight:600; color:{GREEN_D}; line-height:1.2; margin:.25rem 0; }}
    .crop-meta {{ font-size:.93rem; color:#5F6A62; }}
    .bar-bg {{ background:#EBEEE7; border-radius:999px; height:8px; margin-top:.55rem; overflow:hidden; }}
    .bar-fg {{ background:{GREEN_M}; height:8px; border-radius:999px; }}
    .badge {{ display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600; }}
    .b-high {{ background:#E4F0E4; color:#2F6444; }}
    .b-med  {{ background:#F8EFD9; color:#86642A; }}
    .b-low  {{ background:#F6E4DF; color:#8E3B2E; }}
    .why {{ font-size:.97rem; color:#1f2937; padding:.25rem 0; line-height:1.45; }}
    .note {{ background:{EARTH_L}; border-left:3px solid #D8C08A; border-radius:6px;
             padding:.7rem 1rem; font-size:.92rem; color:#5E4F2E; line-height:1.5; }}
    .strip {{ background:#F7F8F4; border:1px solid #E7E9E1; border-radius:8px; padding:.5rem .9rem;
              font-size:.88rem; color:#4F5E54; margin-bottom:1rem; }}
    .changed {{ background:#FBF5EA; border:1px solid #EBD9B6; border-left:4px solid #C79A45;
                border-radius:8px; padding:.7rem 1rem; color:#70521C; }}
    .same {{ background:#F6FAF5; border:1px solid #E1EBDD; border-left:4px solid {GREEN_M};
             border-radius:8px; padding:.7rem 1rem; color:{GREEN_D}; }}
    div[data-testid="stMetric"] {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px;
                                   padding:.75rem 1rem; }}
    div[data-testid="stMetricValue"] {{ color:{GREEN_D}; font-weight:600; }}
    button[data-baseweb="tab"] p {{ font-size:.98rem; font-weight:500; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders (pipeline unchanged)
@st.cache_resource
def load_resources():
    models_dir = os.path.join(BASE_DIR, "models")
    explainer = CropExplainer(models_dir)
    with open(os.path.join(models_dir, "metrics_comparison.json")) as f:
        metrics = json.load(f)
    gp = os.path.join(models_dir, "global_shap_importance.joblib")
    global_shap = joblib.load(gp) if os.path.exists(gp) else None
    cms = joblib.load(os.path.join(models_dir, "confusion_matrices.joblib"))
    return explainer, metrics, global_shap, cms


@st.cache_data
def load_data():
    p = os.path.join(BASE_DIR, "models", "cleaned_dataset.csv")
    if not os.path.exists(p):
        p = os.path.join(BASE_DIR, "data", "crop_recommendation_dataset.csv")
    return pd.read_csv(p)


explainer, metrics_data, global_shap_df, confusion_matrices = load_resources()
df_data = load_data()
CLASSES = [str(c) for c in explainer.classes]
cat = explainer.preprocessor.cat_maps


@st.cache_data(show_spinner=False)
def cached_predict(key: tuple):
    return explainer.predict(dict(key))


@st.cache_data(show_spinner=False)
def cached_shap(key: tuple, class_idx: int, n_bg: int = 40):
    return explainer.explain_instance_shap(dict(key), class_idx, n_bg_samples=n_bg)


def to_key(d):
    return tuple(sorted(d.items()))


def top_n(pred, n=5):
    p = np.asarray(pred["all_probabilities"], dtype=float)
    return [(CLASSES[i], float(p[i])) for i in np.argsort(p)[::-1][:n]]


def prob_of(pred, crop):
    return float(np.asarray(pred["all_probabilities"], dtype=float)[CLASSES.index(crop)])


def badge(p):  # display bands only: <40% low, 40-70% medium, >=70% high
    if p >= 0.70:
        return "High confidence", "b-high"
    if p >= 0.40:
        return "Medium confidence", "b-med"
    return "Low confidence", "b-low"


def style(fig, h=300, legend=False):
    fig.update_layout(height=h, margin=dict(l=0, r=10, t=10, b=0), showlegend=legend,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter", size=13, color="#374151"))
    fig.update_xaxes(gridcolor="#EEF0EA")
    fig.update_yaxes(gridcolor="#EEF0EA")
    return fig


NOBAR = {"displayModeBar": False}


def page_header(title, sub):
    st.markdown(f"<div class='eyebrow'>{APP_NAME.upper()} · TAMIL NADU</div>"
                f"<p class='page-title'>{title}</p><p class='page-sub'>{sub}</p>",
                unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f"<div class='sec-title'>{title}</div>"
                + (f"<div class='sec-sub'>{sub}</div>" if sub else ""), unsafe_allow_html=True)


# ------------------------------------------------------------------ state
RAIN_COL = next((c for c in NUMERICAL_COLS if "RAIN" in c.upper()), None)
soil_opts = sorted(cat["SOIL"].keys())
season_opts = sorted(cat["SEASON"].keys())
water_opts = sorted(cat["WATER_SOURCE"].keys())

NUM = {  # key: (feature, label, min, max, default, step)
    "n":    ("N", "Nitrogen N (kg/ha)", 10.0, 250.0, 85.0, 1.0),
    "p":    ("P", "Phosphorus P (kg/ha)", 5.0, 150.0, 45.0, 1.0),
    "k":    ("K", "Potassium K (kg/ha)", 5.0, 200.0, 45.0, 1.0),
    "ph":   ("SOIL_PH", "Soil pH", 3.5, 10.0, 6.8, 0.1),
    "temp": ("TEMP", "Temperature (°C)", 10.0, 50.0, 26.5, 0.5),
    "hum":  ("RELATIVE_HUMIDITY", "Humidity (%)", 15.0, 100.0, 78.0, 1.0),
}
if RAIN_COL:
    _r = df_data[RAIN_COL]
    NUM["rain"] = (RAIN_COL, "Rainfall", float(_r.min()), float(_r.max()), float(_r.median()), 1.0)

PAGES = ["🏠  Home", "🌾  Get Recommendation", "🧠  Explainable AI", "📊  Visualizations",
         "🔄  What-If Analysis", "🏆  Model Performance", "🗂️  Dataset Explorer"]
P_HOME, P_REC, P_XAI, P_VIZ, P_WI, P_PERF, P_DATA = PAGES

DEFAULTS = {"soil": "alluvial soil" if "alluvial soil" in soil_opts else soil_opts[0],
            "season": "kharif" if "kharif" in season_opts else season_opts[0],
            "water": "irrigated" if "irrigated" in water_opts else water_opts[0],
            "page": P_HOME}
DEFAULTS.update({k: v[4] for k, v in NUM.items()})
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k in list(st.session_state.keys()):      # keep widget values while their page is hidden
    if k in DEFAULTS or k.startswith("wi_"):
        st.session_state[k] = st.session_state[k]


def goto(page):
    st.session_state["page"] = page


def current_inputs():
    d = {NUM[k][0]: float(st.session_state[k]) for k in NUM}
    d.update({"SOIL": st.session_state["soil"], "SEASON": st.session_state["season"],
              "WATER_SOURCE": st.session_state["water"]})
    return d


farmer = current_inputs()
prediction = cached_predict(to_key(farmer))
top_crop, top_prob = prediction["top_crop"], prediction["top_probability"]
top3 = prediction["top_3"]
advisory = explainer.calculate_nutrient_gap_and_advisory(farmer, top_crop)


def field_strip():
    rain = f" · Rainfall {farmer[RAIN_COL]:.0f}" if RAIN_COL else ""
    st.markdown(
        f"<div class='strip'><b>Your field:</b> {farmer['SOIL'].title()} · {farmer['SEASON'].title()} · "
        f"{farmer['WATER_SOURCE'].title()} · N {farmer['N']:.0f} · P {farmer['P']:.0f} · "
        f"K {farmer['K']:.0f} · pH {farmer['SOIL_PH']:.1f} · {farmer['TEMP']:.1f}°C · "
        f"{farmer['RELATIVE_HUMIDITY']:.0f}% humidity{rain}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"<div class='sb-brand'>🌱 {APP_NAME}</div>"
                f"<div class='sb-sub'>Tamil Nadu · Explainable AI</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.markdown("---")
    lab, _ = badge(top_prob)
    st.markdown(f"<div class='sb-now'>Current recommendation<br><b>{top_crop.title()}</b><br>"
                f"{top_prob*100:.1f}% · {lab}</div>", unsafe_allow_html=True)


# ================================================================== HOME
if page == P_HOME:
    ft = metrics_data.get("FT-Transformer", {})
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Tamil Nadu · Explainable AI</div>
        <div class="hero-title">{APP_NAME}</div>
        <div class="hero-tag">{APP_TAGLINE}</div>
        <div class="hero-text">Enter your soil test and weather conditions to get a crop recommendation,
        then see exactly why the model chose it. Built on a transformer model with SHAP-based
        explanations.</div>
        <span class="chip">Explainable AI</span><span class="chip">FT-Transformer</span>
        <span class="chip">SHAP</span><span class="chip">{len(CLASSES)} crops</span>
    </div>""", unsafe_allow_html=True)

    stats = [(f"{len(df_data):,}", "Field records"), (f"{len(CLASSES)}", "Crops covered"),
             (f"{df_data['SOIL'].nunique()}", "Soil types")]
    if "accuracy" in ft:
        stats.append((f"{ft['accuracy']*100:.2f}%", "Held-out test accuracy"))
    for col, (v, l) in zip(st.columns(len(stats)), stats):
        col.markdown(f"<div class='stat'><div class='stat-v'>{v}</div><div class='stat-l'>{l}</div></div>",
                     unsafe_allow_html=True)

    section("Explore the app", "Use the sidebar at any time, or jump straight in.")
    feats = [(P_REC, "🌾 Get Recommendation", "Enter your field conditions and get the best-suited crop with a confidence score."),
             (P_XAI, "🧠 Explainable AI", "See why the model chose this crop: plain-language reasons, SHAP and confidence."),
             (P_VIZ, "📊 Visualizations", "Compare crop suitability, soil nutrients and climate against the crop's needs."),
             (P_WI, "🔄 What-If Analysis", "Change nutrients or weather and watch the recommendation respond."),
             (P_PERF, "🏆 Model Performance", "Benchmarks across five models and the confusion matrix."),
             (P_DATA, "🗂️ Dataset Explorer", "Browse the distributions behind the model.")]
    for row in (feats[:3], feats[3:]):
        for col, (pg, t, d) in zip(st.columns(3), row):
            with col:
                st.markdown(f"<div class='feat'><div class='feat-t'>{t}</div>"
                            f"<div class='feat-d'>{d}</div></div>", unsafe_allow_html=True)
                st.button("Open →", key=f"open_{pg}", on_click=goto, args=(pg,),
                          use_container_width=True)
        st.write("")


# ================================================================== RECOMMENDATION
elif page == P_REC:
    page_header("Get a Recommendation", "Describe your field. The recommendation updates as you adjust.")

    with st.container(border=True):
        st.markdown("**1 · Soil**")
        c = st.columns([1.7, 1, 1, 1, 1])
        c[0].selectbox("Soil type", soil_opts, key="soil")
        for col, k in zip(c[1:], ["n", "p", "k", "ph"]):
            _, lab, lo, hi, _, stp = NUM[k]
            col.slider(lab, lo, hi, step=stp, key=k)
        st.markdown("**2 · Season & weather**")
        keys = ["temp", "hum"] + (["rain"] if RAIN_COL else [])
        c = st.columns(2 + len(keys))
        c[0].selectbox("Season", season_opts, key="season")
        c[1].selectbox("Water source", water_opts, key="water")
        for col, k in zip(c[2:], keys):
            _, lab, lo, hi, _, stp = NUM[k]
            col.slider(lab, lo, hi, step=stp, key=k)

    farmer = current_inputs()
    prediction = cached_predict(to_key(farmer))
    top_crop, top_prob, top3 = prediction["top_crop"], prediction["top_probability"], prediction["top_3"]
    advisory = explainer.calculate_nutrient_gap_and_advisory(farmer, top_crop)

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        lab, cls = badge(top_prob)
        st.markdown(f"""<div class="crop-card">
            <div class="crop-eyebrow">Recommended crop</div>
            <div class="crop-name">{top_crop.title()}</div>
            <div class="crop-meta">{farmer['SEASON'].title()} season · {farmer['WATER_SOURCE'].title()} ·
            {farmer['SOIL'].title()}</div>
            <div style="margin-top:.9rem"><span class="badge {cls}">{lab}</span>
            <span style="font-size:1.4rem;font-weight:600;color:{GREEN_D};margin-left:.5rem">
            {top_prob*100:.1f}%</span></div>
            <div class="bar-bg"><div class="bar-fg" style="width:{top_prob*100:.1f}%"></div></div>
            <div class="crop-meta" style="margin-top:.35rem">Model confidence (softmax probability)</div>
            </div>""", unsafe_allow_html=True)
        st.write("")
        if len(top3) > 1:
            st.markdown("**Other good options**")
            for col, r in zip(st.columns(len(top3) - 1), top3[1:]):
                col.metric(r["crop"].title(), f"{r['probability']*100:.1f}%")
        st.button("See full explanation →", on_click=goto, args=(P_XAI,))

    with right:
        st.markdown("**Why this crop**")
        st.markdown(f"<div class='why'>✅ Suits the <b>{farmer['SEASON'].title()}</b> season with "
                    f"<b>{farmer['WATER_SOURCE']}</b> water.</div>", unsafe_allow_html=True)
        if advisory:
            if abs(advisory.get("delta_ph", 0)) <= 0.5:
                st.markdown(f"<div class='why'>✅ Soil pH {farmer['SOIL_PH']:.1f} is close to what "
                            f"{top_crop.title()} needs ({advisory['target_ph']:.1f}).</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='why'>⚠️ Soil pH is {farmer['SOIL_PH']:.1f}; "
                            f"{top_crop.title()} prefers about {advisory['target_ph']:.1f}.</div>",
                            unsafe_allow_html=True)
            for lbl, tk, dk in [("Nitrogen", "target_n", "delta_n"),
                                ("Phosphorus", "target_p", "delta_p"),
                                ("Potassium", "target_k", "delta_k")]:
                tgt, d = advisory.get(tk, 0), advisory.get(dk, 0)
                if abs(d) <= max(0.2 * tgt, 5):
                    msg = f"✅ {lbl} is close to the benchmark."
                elif d > 0:
                    msg = f"⚠️ {lbl} is low by about {d:.0f} kg/ha."
                else:
                    msg = f"✅ {lbl} is above the benchmark; no extra needed."
                st.markdown(f"<div class='why'>{msg}</div>", unsafe_allow_html=True)
        st.markdown("**What to do**")
        notes = (advisory or {}).get("advisory_notes") or []
        if notes:
            for n_ in notes[:4]:
                st.markdown(n_)
        else:
            st.write("No soil corrections needed for these values.")


# ================================================================== EXPLAINABLE AI
elif page == P_XAI:
    page_header("Explainable AI", f"Why the model recommends {top_crop.title()} for your field.")
    field_strip()

    gap = (top_prob - top3[1]["probability"]) * 100 if len(top3) > 1 else None
    lab, cls = badge(top_prob)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended crop", top_crop.title())
    m2.metric("Model confidence", f"{top_prob*100:.1f}%")
    m3.metric("Lead over runner-up", f"{gap:.1f} pts" if gap is not None else "–")
    m4.metric("Confidence level", lab.replace(" confidence", ""))

    with st.spinner("Computing SHAP attributions for your field..."):
        shap_res = cached_shap(to_key(farmer), top3[0]["class_index"])
    sv = shap_res["shap_values"]
    sdf = pd.DataFrame([{"Feature": k, "SHAP value": v,
                         "Effect": "Supports this crop" if v > 0 else "Works against this crop"}
                        for k, v in sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)])

    t_plain, t_shap, t_attn, t_glob, t_conf = st.tabs(
        ["In plain words", "SHAP · this field", "Attention", "Global importance", "Confidence"])

    with t_plain:
        section("The explanation", "Written for farmers; derived from the SHAP values and soil benchmarks.")
        st.markdown(explainer.generate_human_explanation(farmer, prediction, shap_res))
        st.markdown("**Strongest influences on this recommendation**")
        for _, r in sdf.head(3).iterrows():
            icon = "🟢" if r["SHAP value"] > 0 else "🔴"
            st.markdown(f"<div class='why'>{icon} <b>{r['Feature']}</b> – {r['Effect'].lower()} "
                        f"({r['SHAP value']:+.3f})</div>", unsafe_allow_html=True)
        notes = (advisory or {}).get("advisory_notes") or []
        if notes:
            st.markdown("**Recommended soil actions**")
            for n_ in notes:
                st.markdown(n_)

    with t_shap:
        st.markdown("<div class='note'><b>How to read this:</b> each bar is how much that input pushed "
                    "the model's probability for the recommended crop <b>up (green)</b> or <b>down "
                    "(red)</b>, compared with an average field. Longer bar = bigger influence.</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fs = px.bar(sdf, x="SHAP value", y="Feature", orientation="h", color="Effect",
                        color_discrete_map={"Supports this crop": GREEN,
                                            "Works against this crop": RED})
            fs.update_yaxes(autorange="reversed", title=None)
            fs.update_layout(legend=dict(orientation="h", y=-0.15, title=None))
            st.plotly_chart(style(fs, 360, True), use_container_width=True, config=NOBAR)
        with c2:
            st.dataframe(sdf[["Feature", "SHAP value"]].style.format({"SHAP value": "{:+.4f}"}),
                         hide_index=True, use_container_width=True, height=360)
        base_val = shap_res.get("base_value", 0.0)
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute"] + ["relative"] * len(sdf) + ["total"],
            x=["Average field"] + sdf["Feature"].tolist() + ["Final probability"],
            y=[base_val] + sdf["SHAP value"].tolist() + [0.0],
            connector={"line": {"color": "#9ca3af"}},
            decreasing={"marker": {"color": RED}}, increasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": GREEN_D}}))
        section("Step by step", "Starting from an average field, each input adds or removes probability.")
        st.plotly_chart(style(wf, 340), use_container_width=True, config=NOBAR)

    with t_attn:
        st.markdown("<div class='note'><b>What this shows:</b> the share of the transformer's internal "
                    "attention given to each input for this field. It tells you where the model "
                    "<i>focused</i>; it does not prove cause, so use SHAP for influence.</div>",
                    unsafe_allow_html=True)
        ad = pd.DataFrame([{"Feature": k, "Attention share (%)": v}
                           for k, v in sorted(prediction["attention_weights"].items(),
                                              key=lambda x: x[1], reverse=True)])
        fa = px.bar(ad, x="Attention share (%)", y="Feature", orientation="h",
                    color_discrete_sequence=[GREEN])
        fa.update_yaxes(autorange="reversed", title=None)
        st.plotly_chart(style(fa, 340), use_container_width=True, config=NOBAR)

    with t_glob:
        st.markdown("<div class='note'><b>What this shows:</b> mean |SHAP| across the whole test set, "
                    "i.e. how much the model relies on each input overall, for all "
                    f"{len(CLASSES)} crops.</div>", unsafe_allow_html=True)
        if global_shap_df is not None:
            fg = px.bar(global_shap_df, x="mean_abs_shap", y="feature", orientation="h",
                        color_discrete_sequence=[GREEN_D],
                        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""})
            fg.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fg, 340), use_container_width=True, config=NOBAR)
        else:
            st.info("global_shap_importance.joblib was not found in models/.")

    with t_conf:
        st.markdown("<div class='note'><b>What this shows:</b> the model's probability for each crop "
                    "(they sum to 100% over all crops). A high top score with a clear gap to the "
                    "runner-up means a decisive recommendation. This is model confidence, not a "
                    "guarantee of yield.</div>", unsafe_allow_html=True)
        g1, g2 = st.columns([1, 1.4])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=top_prob * 100, number={"suffix": "%", "valueformat": ".1f"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN},
                       "steps": [{"range": [0, 40], "color": "#F3DDD7"},
                                 {"range": [40, 70], "color": "#F5EAC8"},
                                 {"range": [70, 100], "color": "#DCEBDD"}]}))
            st.plotly_chart(style(gauge, 260), use_container_width=True, config=NOBAR)
            st.caption("Bands (<40 low, 40–70 medium, ≥70 high) are display thresholds only.")
        with g2:
            t5 = pd.DataFrame(top_n(prediction, 5), columns=["Crop", "p"])
            t5["Crop"] = t5["Crop"].str.title()
            t5["Probability (%)"] = (t5["p"] * 100).round(2)
            fc = px.bar(t5, x="Probability (%)", y="Crop", orientation="h", text="Probability (%)")
            fc.update_traces(marker_color=[GREEN] + [GREEN_M] * 4, textposition="outside",
                             cliponaxis=False)
            fc.update_yaxes(autorange="reversed", title=None)
            fc.update_xaxes(range=[0, 108])
            st.plotly_chart(style(fc, 260), use_container_width=True, config=NOBAR)

    st.download_button("⬇ Download full advisory report (.txt)",
                       explainer.generate_full_dossier(farmer, prediction, shap_res),
                       file_name=f"crop_advisory_{top_crop.lower().replace(' ', '_')}.txt",
                       mime="text/plain")


# ================================================================== VISUALIZATIONS
elif page == P_VIZ:
    page_header("Visualizations", f"How your field compares with what {top_crop.title()} needs.")
    field_strip()

    c1, c2 = st.columns(2)
    with c1:
        section("Crop suitability", "Model probability for the five best-matching crops")
        t5 = pd.DataFrame(top_n(prediction, 5), columns=["Crop", "p"])
        t5["Crop"] = t5["Crop"].str.title()
        t5["Match (%)"] = (t5["p"] * 100).round(1)
        fig = px.bar(t5, x="Match (%)", y="Crop", orientation="h", text="Match (%)")
        fig.update_traces(marker_color=[GREEN] + [GREEN_L] * 4, marker_line_color=GREEN,
                          marker_line_width=1, textposition="outside", cliponaxis=False)
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(range=[0, 108], title="Probability (%)")
        st.plotly_chart(style(fig, 300), use_container_width=True, config=NOBAR)
    with c2:
        section("Soil nutrients", f"Your soil vs the {top_crop.title()} benchmark (kg/ha)")
        if advisory and all(k in advisory for k in ("target_n", "target_p", "target_k")):
            nd = pd.DataFrame({
                "Nutrient": ["N", "P", "K"] * 2,
                "Series": ["Your soil"] * 3 + ["Crop benchmark"] * 3,
                "kg/ha": [farmer["N"], farmer["P"], farmer["K"],
                          advisory["target_n"], advisory["target_p"], advisory["target_k"]]})
            fn = px.bar(nd, x="Nutrient", y="kg/ha", color="Series", barmode="group",
                        color_discrete_map={"Your soil": GREEN, "Crop benchmark": EARTH})
            fn.update_xaxes(title=None)
            fn.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
            st.plotly_chart(style(fn, 300, True), use_container_width=True, config=NOBAR)
        else:
            st.info("No nutrient benchmark available for this crop.")

    section("Environmental conditions",
            f"Shaded band = middle 80% of dataset records for {top_crop.title()}. "
            "Diamond = your value (dark green: inside, orange: outside).")
    sub = df_data[df_data["CROPS"].astype(str).str.lower() == top_crop.lower()]
    specs = [("TEMP", "Temperature (°C)", farmer["TEMP"]),
             ("RELATIVE_HUMIDITY", "Humidity (%)", farmer["RELATIVE_HUMIDITY"]),
             ("SOIL_PH", "Soil pH", farmer["SOIL_PH"])]
    if RAIN_COL:
        specs.append((RAIN_COL, "Rainfall", farmer[RAIN_COL]))
    for col, (name, title, val) in zip(st.columns(len(specs)), specs):
        if sub.empty or name not in sub:
            col.caption(f"{title}: no dataset records for this crop.")
            continue
        lo, hi = float(sub[name].quantile(.10)), float(sub[name].quantile(.90))
        amin, amax = float(df_data[name].min()), float(df_data[name].max())
        inside = lo <= val <= hi
        fg = go.Figure()
        fg.add_shape(type="rect", x0=amin, x1=amax, y0=-.25, y1=.25, fillcolor="#EDEFE9", line_width=0)
        fg.add_shape(type="rect", x0=lo, x1=hi, y0=-.25, y1=.25, fillcolor="#CFE3D0", line_width=0)
        fg.add_trace(go.Scatter(x=[val], y=[0], mode="markers",
                                marker=dict(symbol="diamond", size=16,
                                            color=GREEN_D if inside else "#C79A45"),
                                hovertemplate=f"{title}: %{{x:.1f}}<extra></extra>"))
        fg.update_yaxes(visible=False, range=[-1, 1])
        fg.update_xaxes(range=[amin, amax])
        style(fg, 130)
        fg.update_layout(margin=dict(l=0, r=10, t=36, b=0),
                         title=dict(text=f"{title}: <b>{val:.1f}</b><br>"
                                         f"<span style='font-size:11px;color:#6b7280'>"
                                         f"{'In range' if inside else 'Outside range'} "
                                         f"({lo:.1f}–{hi:.1f})</span>", font=dict(size=13), x=0))
        col.plotly_chart(fg, use_container_width=True, config=NOBAR)


# ================================================================== WHAT-IF
elif page == P_WI:
    page_header("What-If Analysis", "Change conditions and see how the recommendation responds.")
    field_strip()
    base = farmer
    wi_map = {k: f"wi_{k}" for k in NUM}
    if st.session_state.get("wi_sig") != to_key(base):
        for k, wk in wi_map.items():
            st.session_state[wk] = float(base[NUM[k][0]])
        st.session_state["wi_sig"] = to_key(base)

    def reset_whatif():
        for k, wk in wi_map.items():
            st.session_state[wk] = float(base[NUM[k][0]])

    ctrl, out = st.columns([1, 1.55])
    with ctrl:
        section("Adjust conditions", "Soil type, season and water source stay as set on the recommendation page.")
        a, b = st.columns(2)
        order = ["n", "p", "k", "ph", "temp", "hum"] + (["rain"] if RAIN_COL else [])
        for i, k in enumerate(order):
            _, lab, lo, hi, _, stp = NUM[k]
            (a if i % 2 == 0 else b).slider(lab, lo, hi, step=stp, key=wi_map[k])
        st.button("↺ Reset to original", on_click=reset_whatif, use_container_width=True)
        if not RAIN_COL:
            st.markdown("<div class='note'>Rainfall is not an input of the trained model "
                        "(N, P, K, pH, temperature, humidity, soil, season, water source), so it "
                        "cannot change a prediction and is not offered. Humidity is the model's "
                        "moisture input.</div>", unsafe_allow_html=True)

    mod = {**base, **{NUM[k][0]: float(st.session_state[wi_map[k]]) for k in NUM}}
    mpred = cached_predict(to_key(mod))
    m_crop, m_prob = mpred["top_crop"], mpred["top_probability"]

    with out:
        section("Original vs modified")
        o1, o2 = st.columns(2)
        for colx, title, crop, prob, same in [(o1, "ORIGINAL", top_crop, top_prob, True),
                                              (o2, "MODIFIED", m_crop, m_prob, m_crop == top_crop)]:
            colour = GREEN if same else "#C79A45"
            colx.markdown(f"""<div class="crop-card" style="border-left-color:{colour}">
                <div class="crop-eyebrow" style="color:{colour}">{title}</div>
                <div class="crop-name" style="font-size:1.8rem;color:{colour}">{crop.title()}</div>
                <div class="crop-meta">Probability <b>{prob*100:.1f}%</b></div></div>""",
                          unsafe_allow_html=True)
        st.write("")
        if m_crop != top_crop:
            st.markdown(f"<div class='changed'>The recommendation <b>changed</b>: "
                        f"{top_crop.title()} → <b>{m_crop.title()}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='same'>Same crop recommended: {top_crop.title()} "
                        f"({(m_prob - top_prob)*100:+.1f} pts).</div>", unsafe_allow_html=True)

        crops = list(dict.fromkeys([c_ for c_, _ in top_n(prediction, 5)] +
                                   [c_ for c_, _ in top_n(mpred, 5)]))
        cmp_df = pd.DataFrame(
            [{"Crop": c_.title(), "Scenario": "Original", "Probability (%)": prob_of(prediction, c_) * 100}
             for c_ in crops] +
            [{"Crop": c_.title(), "Scenario": "Modified", "Probability (%)": prob_of(mpred, c_) * 100}
             for c_ in crops])
        fc = px.bar(cmp_df, x="Crop", y="Probability (%)", color="Scenario", barmode="group",
                    color_discrete_map={"Original": GREY, "Modified": GREEN})
        fc.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=None))
        st.plotly_chart(style(fc, 300, True), use_container_width=True, config=NOBAR)

    diffs = [{"Input": NUM[k][1], "Original": base[NUM[k][0]], "Modified": mod[NUM[k][0]],
              "Change": mod[NUM[k][0]] - base[NUM[k][0]]}
             for k in NUM if abs(mod[NUM[k][0]] - base[NUM[k][0]]) > 1e-9]
    if diffs:
        section("What you changed")
        st.dataframe(pd.DataFrame(diffs).style.format(
            {"Original": "{:.1f}", "Modified": "{:.1f}", "Change": "{:+.1f}"}),
            hide_index=True, use_container_width=True)


# ================================================================== MODEL PERFORMANCE
elif page == P_PERF:
    page_header("Model Performance", "Results on the held-out test set, taken from the saved evaluation files.")
    ft = metrics_data.get("FT-Transformer")
    if ft:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{ft['accuracy']*100:.2f}%")
        m2.metric("Macro F1", f"{ft['f1_macro']*100:.2f}%")
        m3.metric("Macro precision", f"{ft['precision_macro']*100:.2f}%")
        m4.metric("Macro recall", f"{ft['recall_macro']*100:.2f}%")
        st.caption("FT-Transformer (primary model)")

    t_b, t_c, t_a = st.tabs(["Model comparison", "Confusion matrix", "Architecture"])
    with t_b:
        rows = [{"Model": n, "Accuracy": m["accuracy"], "Macro F1": m["f1_macro"],
                 "Macro Precision": m["precision_macro"], "Macro Recall": m["recall_macro"],
                 "Train time (s)": m["train_time_sec"]} for n, m in metrics_data.items()]
        dfb = pd.DataFrame(rows)
        fmt = {c: "{:.4f}" for c in dfb.columns if c not in ("Model", "Train time (s)")}
        fmt["Train time (s)"] = "{:.2f}"
        fb = px.bar(dfb, x="Model", y="Accuracy", text="Accuracy", color_discrete_sequence=[GREEN])
        fb.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fb.update_yaxes(range=[max(0, dfb["Accuracy"].min() - 0.02), min(1.0, dfb["Accuracy"].max() + 0.01)])
        st.plotly_chart(style(fb, 320), use_container_width=True, config=NOBAR)
        st.dataframe(dfb.style.format(fmt), hide_index=True, use_container_width=True)

    with t_c:
        names = list(confusion_matrices.keys())
        cA, cB = st.columns([1, 1])
        sel = cA.selectbox("Model", names, index=names.index("FT-Transformer") if "FT-Transformer" in names else 0)
        norm = cB.toggle("Normalise rows (share of each true crop)", value=True)
        cm = np.array(confusion_matrices[sel])
        st.caption(f"{sel}: {int(np.trace(cm))} of {int(cm.sum())} test samples classified correctly "
                   f"({np.trace(cm)/cm.sum()*100:.2f}%). Bright diagonal = correct predictions.")
        z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if norm else cm
        labels = CLASSES if len(CLASSES) == cm.shape[0] else list(range(cm.shape[0]))
        fcm = px.imshow(z, x=labels, y=labels, color_continuous_scale="Greens",
                        labels=dict(x="Predicted", y="Actual", color="Share" if norm else "Count"))
        fcm.update_xaxes(tickfont=dict(size=8), tickangle=90)
        fcm.update_yaxes(tickfont=dict(size=8))
        st.plotly_chart(style(fcm, 640), use_container_width=True)
        off = cm.astype(float).copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(off.ravel())[::-1][:10]
        pairs = [{"Actual crop": str(labels[i // cm.shape[1]]).title(),
                  "Predicted as": str(labels[i % cm.shape[1]]).title(),
                  "Samples": int(off.ravel()[i])} for i in flat if off.ravel()[i] > 0]
        if pairs:
            section("Most frequent confusions")
            st.dataframe(pd.DataFrame(pairs), hide_index=True, use_container_width=True)

    with t_a:
        st.markdown("""
```
Input features
├── 6 numerical (N, P, K, pH, temperature, humidity)  → linear tokenizer
├── 3 categorical (soil, season, water source)        → embeddings
└── learnable [CLS] token
          │
Transformer encoder (multi-head self-attention → attention weights used for XAI)
          │
[CLS] representation → LayerNorm → linear head → softmax over crops
```""")


# ================================================================== DATASET EXPLORER
elif page == P_DATA:
    page_header("Dataset Explorer", f"{len(df_data):,} records across {df_data['CROPS'].nunique()} crops.")
    d1, d2, d3, d4 = st.tabs(["Distributions", "Climate by season", "Composition", "Correlation"])

    with d1:
        a, b = st.columns([1, 3])
        crop_f = a.selectbox("Crop", ["All crops"] + sorted(df_data["CROPS"].astype(str).unique()))
        feat = a.selectbox("Feature", [c for c in ["N", "P", "K", "SOIL_PH", "TEMP", "RELATIVE_HUMIDITY"]
                                       if c in df_data.columns])
        fdf = df_data if crop_f == "All crops" else df_data[df_data["CROPS"] == crop_f]
        b.plotly_chart(style(px.histogram(fdf, x=feat, nbins=30, color_discrete_sequence=[GREEN]), 340),
                       use_container_width=True, config=NOBAR)

    with d2:
        e1, e2 = st.columns(2)
        pal = [GREEN, EARTH, GREEN_M]
        for colx, f_, t_ in [(e1, "TEMP", "Temperature (°C)"), (e2, "RELATIVE_HUMIDITY", "Humidity (%)")]:
            colx.markdown(f"**{t_}**")
            colx.plotly_chart(style(px.box(df_data, x="SEASON", y=f_, color="SEASON",
                                           color_discrete_sequence=pal, labels={f_: t_}), 340),
                              use_container_width=True, config=NOBAR)

    with d3:
        f1, f2, f3 = st.columns(3)
        sc = df_data["SEASON"].value_counts().reset_index()
        sc.columns = ["Season", "Records"]
        f1.markdown("**By season**")
        f1.plotly_chart(style(px.pie(sc, names="Season", values="Records", hole=.45,
                                     color_discrete_sequence=[GREEN, EARTH, GREEN_M]), 290, True),
                        use_container_width=True, config=NOBAR)
        wc = df_data["WATER_SOURCE"].value_counts().reset_index()
        wc.columns = ["Water source", "Records"]
        f2.markdown("**By water source**")
        f2.plotly_chart(style(px.pie(wc, names="Water source", values="Records", hole=.45,
                                     color_discrete_sequence=[GREEN, EARTH]), 290, True),
                        use_container_width=True, config=NOBAR)
        so = df_data["SOIL"].value_counts().head(8).reset_index()
        so.columns = ["Soil", "Records"]
        fso = px.bar(so, x="Records", y="Soil", orientation="h", color_discrete_sequence=[GREEN])
        fso.update_yaxes(autorange="reversed", title=None)
        f3.markdown("**Top soil types**")
        f3.plotly_chart(style(fso, 290), use_container_width=True, config=NOBAR)

    with d4:
        corr = df_data[[c for c in NUMERICAL_COLS if c in df_data.columns]].corr()
        st.plotly_chart(style(px.imshow(corr, text_auto=".2f", color_continuous_scale="Greens",
                                        aspect="auto"), 420), use_container_width=True)
