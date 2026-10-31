# 🩺 Diabetes Progression Predictor

A Streamlit web app that predicts **diabetes disease progression one year after baseline** using a regression model trained on the sklearn Diabetes dataset (442 patients, 10 features).

## Features

- Loads a pre-trained model (`best_diabetes_model.joblib`)
- Clean, sectioned input form (demographics, vitals, blood serum tests)
- Inline tooltips and a reference table explaining every feature (including `s1`–`s6`)
- One-click prediction with a qualitative risk level (Low / Moderate / High)
- Model coefficient viewer for interpretability

## The 10 input features

| Feature | Meaning |
|---------|---------|
| `age` | Patient age (years) |
| `sex` | Patient sex |
| `bmi` | Body mass index (weight / height²) |
| `bp` | Mean arterial blood pressure (mm Hg) |
| `s1` | Total serum cholesterol (tc) |
| `s2` | Low-density lipoproteins — "bad" cholesterol (ldl) |
| `s3` | High-density lipoproteins — "good" cholesterol (hdl) |
| `s4` | Total cholesterol / HDL ratio (tch) |
| `s5` | Log of serum triglycerides (ltg) |
| `s6` | Blood sugar level (glu) |

> All features are **sklearn-normalized**: `(value − mean) ÷ std`.

The **target** is a quantitative measure of disease progression one year after baseline.

## Project structure

```
P6/
├── app.py                          # Streamlit application
├── best_diabetes_model.joblib      # Trained regression model
├── requirements.txt                # Python dependencies
├── diabetes_regression_project.ipynb  # Model training notebook
└── README.md
```

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Dependencies

- streamlit
- scikit-learn
- joblib
- numpy
- pandas

## Usage

1. Enter the patient's baseline measurements in the input fields.
2. Click **Predict**.
3. Read the predicted progression score and risk level.

## License

For educational use (HCCDA-AI course project).