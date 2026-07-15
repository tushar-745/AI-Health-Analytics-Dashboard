# 🏥 HealthAI Dashboard

A Machine Learning–powered Healthcare Analytics Dashboard built with Python & Streamlit.

---

## 📋 Features

| Page | Description |
|---|---|
| 📊 Overview | Hospital KPIs, department load, patient status, admission trends |
| 🫀 Heart Disease Predictor | Gradient Boosting model — enter vitals, get risk % + feature importance |
| 🩸 Diabetes Predictor | Random Forest model — enter measurements, get risk % + feature importance |
| 🏨 Patient Analytics | Filterable patient table, cost analysis, satisfaction, LOS violin plots |
| 📈 Model Performance | Accuracy, AUC, feature importances, data distribution charts |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data & train models (first time only)
```bash
python generate_data.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🤖 ML Models

### Diabetes Prediction
- **Algorithm:** Random Forest Classifier (100 trees)
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Accuracy:** ~85% | **AUC:** ~0.92

### Heart Disease Prediction
- **Algorithm:** Gradient Boosting Classifier (100 trees)
- **Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting BS, ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Major Vessels, Thalassemia
- **Accuracy:** ~84% | **AUC:** ~0.91

---

## 📁 Project Structure
```
healthcare_dashboard/
├── app.py               ← Main Streamlit app
├── generate_data.py     ← Data generation + model training script
├── requirements.txt     ← Python dependencies
├── README.md
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── patients.csv
└── models/
    ├── diabetes_model.pkl
    ├── diabetes_scaler.pkl
    ├── diabetes_features.pkl
    ├── heart_model.pkl
    ├── heart_scaler.pkl
    ├── heart_features.pkl
    └── metrics.pkl
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML:** Scikit-learn (Random Forest, Gradient Boosting)
- **Data:** Pandas, NumPy
- **Styling:** Custom CSS (Inter + DM Serif Display fonts)

---

## 📸 Pages Preview

1. **Overview** — KPI cards, department bar chart, status donut, age histogram, monthly trend line
2. **Heart Predictor** — 13-feature input form, risk gauge, feature importance bars
3. **Diabetes Predictor** — 8-feature sliders, risk gauge, scatter plot context
4. **Patient Analytics** — Filters, cost by department, violin LOS, satisfaction chart, data table
5. **Model Performance** — Side-by-side feature importances, distribution histograms, model summary table
