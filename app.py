import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("best_diabetes_model.joblib")


FEATURES = [
    ("age", "Age", "Patient age (years)"),
    ("sex", "Sex", "Patient sex"),
    ("bmi", "BMI", "Body mass index = weight / height²"),
    ("bp", "Blood Pressure", "Mean arterial blood pressure (mm Hg)"),
    ("s1", "s1 — Total Cholesterol", "Total serum cholesterol (tc)"),
    ("s2", "s2 — LDL", "Low-density lipoproteins — 'bad' cholesterol"),
    ("s3", "s3 — HDL", "High-density lipoproteins — 'good' cholesterol"),
    ("s4", "s4 — TC/HDL Ratio", "Total cholesterol to HDL ratio (tch)"),
    ("s5", "s5 — Log Triglycerides", "Log of serum triglycerides level (ltg)"),
    ("s6", "s6 — Glucose", "Blood sugar level (glu)"),
]

DEFAULTS = {
    "age": 0.05, "sex": 0.05, "bmi": 0.06, "bp": 0.02,
    "s1": 0.04, "s2": -0.04, "s3": -0.03, "s4": -0.004,
    "s5": 0.02, "s6": 0.04,
}

model = load_model()
feature_keys = [f[0] for f in FEATURES]

with st.sidebar:
    st.header("🩺 About this app")
    st.write(
        "Predicts diabetes disease progression one year after baseline, "
        "using a regression model trained on the sklearn Diabetes dataset "
        "(442 patients, 10 features)."
    )
    st.header("📋 Feature reference")
    for _, label, desc in FEATURES:
        st.markdown(f"**{label}** — {desc}")
    st.info("All inputs are sklearn-normalized: (value − mean) ÷ std.")

st.title("Diabetes Progression Predictor")
st.write("Enter the patient's baseline measurements, then click **Predict**.")

with st.expander("ℹ️ What do s1–s6 mean?", expanded=False):
    st.markdown(
        """
| Code | Medical term | Meaning |
|------|--------------|---------|
| **s1** | tc | Total serum cholesterol |
| **s2** | ldl | Low-density lipoproteins — **bad** cholesterol |
| **s3** | hdl | High-density lipoproteins — **good** cholesterol |
| **s4** | tch | Total cholesterol / HDL ratio |
| **s5** | ltg | Log of serum triglycerides (blood fat) |
| **s6** | glu | Blood sugar (glucose) |
    """
    )

st.subheader("Demographics & vitals")
c1, c2, c3, c4 = st.columns(4)
inputs = {}
with c1:
    inputs["age"] = st.number_input("Age", value=DEFAULTS["age"], help="Patient age (years), normalized")
with c2:
    inputs["sex"] = st.number_input("Sex", value=DEFAULTS["sex"], help="Patient sex, normalized")
with c3:
    inputs["bmi"] = st.number_input("BMI", value=DEFAULTS["bmi"], help="Body mass index, normalized")
with c4:
    inputs["bp"] = st.number_input("Blood Pressure", value=DEFAULTS["bp"], help="Mean blood pressure (mm Hg), normalized")

st.subheader("Blood serum measurements (s1–s6)")
c1, c2, c3 = st.columns(3)
with c1:
    inputs["s1"] = st.number_input("s1 — Total Cholesterol", value=DEFAULTS["s1"], help="Total serum cholesterol (tc)")
    inputs["s2"] = st.number_input("s2 — LDL", value=DEFAULTS["s2"], help="Low-density lipoproteins — 'bad' cholesterol")
with c2:
    inputs["s3"] = st.number_input("s3 — HDL", value=DEFAULTS["s3"], help="High-density lipoproteins — 'good' cholesterol")
    inputs["s4"] = st.number_input("s4 — TC/HDL Ratio", value=DEFAULTS["s4"], help="Total cholesterol / HDL ratio")
with c3:
    inputs["s5"] = st.number_input("s5 — Log Triglycerides", value=DEFAULTS["s5"], help="Log of serum triglycerides (ltg)")
    inputs["s6"] = st.number_input("s6 — Glucose", value=DEFAULTS["s6"], help="Blood sugar level (glu)")

feature_df = pd.DataFrame([inputs], columns=feature_keys)
with st.expander("Feature summary", expanded=False):
    st.dataframe(feature_df, use_container_width=True)

st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    prediction = float(model.predict(feature_df)[0])
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Predicted progression", f"{prediction:.2f}")
    with col_b:
        if prediction < 100:
            level, msg = "Low", "Low predicted progression over the next year."
        elif prediction < 200:
            level, msg = "Moderate", "Moderate predicted progression — monitor the patient."
        else:
            level, msg = "High", "High predicted progression — consider early intervention."
        st.success(f"**Risk level: {level}** — {msg}")

    with st.expander("Model coefficients (feature importance)"):
        if hasattr(model, "coef_"):
            coefs = model.coef_
            imp = pd.DataFrame({"Feature": feature_keys, "Coefficient": coefs})
            imp = imp.reindex(imp.Coefficient.abs().sort_values(ascending=False).index)
            st.dataframe(imp, use_container_width=True)
        else:
            st.write("This model type does not expose linear coefficients.")
