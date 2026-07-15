"""
Run this script once to generate synthetic data and train ML models.
    python generate_data.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
N = 1000

print("Generating datasets...")

# ── DIABETES ──────────────────────────────────────────────────
age        = np.random.randint(20, 80, N)
bmi        = np.round(np.random.normal(28, 6, N).clip(15, 55), 1)
glucose    = np.random.randint(70, 200, N)
bp         = np.random.randint(60, 130, N)
insulin    = np.random.randint(0, 300, N)
skin_thick = np.random.randint(10, 60, N)
pregnancies= np.random.randint(0, 15, N)
dpf        = np.round(np.random.uniform(0.08, 2.5, N), 3)

score = (
    (glucose > 140).astype(int) * 3 +
    (bmi > 30).astype(int) * 2 +
    (age > 50).astype(int) * 1.5 +
    (bp > 90).astype(int) * 1 +
    (insulin > 150).astype(int) * 1 +
    (dpf > 1.0).astype(int) * 1
)
diabetes = (score + np.random.normal(0, 1, N) > 4).astype(int)

df_diabetes = pd.DataFrame({
    "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
    "SkinThickness": skin_thick, "Insulin": insulin, "BMI": bmi,
    "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": diabetes
})

# ── HEART ─────────────────────────────────────────────────────
age2     = np.random.randint(29, 77, N)
sex      = np.random.randint(0, 2, N)
cp       = np.random.randint(0, 4, N)
trestbps = np.random.randint(90, 200, N)
chol     = np.random.randint(150, 400, N)
fbs      = (np.random.random(N) > 0.85).astype(int)
restecg  = np.random.randint(0, 3, N)
thalach  = np.random.randint(70, 200, N)
exang    = np.random.randint(0, 2, N)
oldpeak  = np.round(np.random.uniform(0, 6, N), 1)
slope    = np.random.randint(0, 3, N)
ca       = np.random.randint(0, 4, N)
thal     = np.random.randint(0, 4, N)

h_score = (
    (age2 > 55).astype(int) * 2 +
    (sex == 1).astype(int) * 1 +
    (cp > 1).astype(int) * 2 +
    (trestbps > 140).astype(int) * 1 +
    (chol > 250).astype(int) * 1 +
    (exang == 1).astype(int) * 2 +
    (oldpeak > 2).astype(int) * 2
)
heart = (h_score + np.random.normal(0, 1.5, N) > 5).astype(int)

df_heart = pd.DataFrame({
    "age": age2, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": heart
})

# ── PATIENTS ──────────────────────────────────────────────────
genders = np.random.choice(["Male", "Female"], N)
names   = [f"Patient_{i+1:04d}" for i in range(N)]
dates   = pd.date_range("2022-01-01", periods=N, freq="8h")
dept    = np.random.choice(["Cardiology","Neurology","Orthopedics","General","Pediatrics","Oncology"], N)
status  = np.random.choice(["Admitted","Discharged","Critical","Stable"], N, p=[0.3,0.4,0.1,0.2])
los     = np.random.randint(1, 30, N)
charges = np.round(np.random.uniform(5000, 250000, N), 2)
satisfaction = np.random.randint(1, 6, N)

df_patients = pd.DataFrame({
    "PatientID": names, "Gender": genders, "Age": age,
    "Department": dept, "Status": status, "AdmissionDate": dates,
    "LengthOfStay": los, "TotalCharges": charges,
    "Satisfaction": satisfaction, "BMI": bmi, "Glucose": glucose,
    "BloodPressure": bp, "Cholesterol": chol[:N]
})

df_diabetes.to_csv("data/diabetes.csv",  index=False)
df_heart.to_csv("data/heart.csv",        index=False)
df_patients.to_csv("data/patients.csv",  index=False)
print("  ✓ Datasets saved to data/")

# ── TRAIN MODELS ──────────────────────────────────────────────
print("Training models...")

scaler_d = StandardScaler()
X_d = df_diabetes.drop("Outcome", axis=1)
y_d = df_diabetes["Outcome"]
X_d_s = scaler_d.fit_transform(X_d)
X_tr, X_te, y_tr, y_te = train_test_split(X_d_s, y_d, test_size=0.2, random_state=42)
rf_d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_d.fit(X_tr, y_tr)
acc_d = accuracy_score(y_te, rf_d.predict(X_te))
auc_d = roc_auc_score(y_te, rf_d.predict_proba(X_te)[:,1])
print(f"  ✓ Diabetes   → Accuracy: {acc_d:.3f}  AUC: {auc_d:.3f}")

scaler_h = StandardScaler()
X_h = df_heart.drop("target", axis=1)
y_h = df_heart["target"]
X_h_s = scaler_h.fit_transform(X_h)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_h_s, y_h, test_size=0.2, random_state=42)
rf_h = GradientBoostingClassifier(n_estimators=100, random_state=42)
rf_h.fit(X_tr2, y_tr2)
acc_h = accuracy_score(y_te2, rf_h.predict(X_te2))
auc_h = roc_auc_score(y_te2, rf_h.predict_proba(X_te2)[:,1])
print(f"  ✓ Heart      → Accuracy: {acc_h:.3f}  AUC: {auc_h:.3f}")

with open("models/diabetes_model.pkl",    "wb") as f: pickle.dump(rf_d,     f)
with open("models/diabetes_scaler.pkl",   "wb") as f: pickle.dump(scaler_d, f)
with open("models/heart_model.pkl",       "wb") as f: pickle.dump(rf_h,     f)
with open("models/heart_scaler.pkl",      "wb") as f: pickle.dump(scaler_h, f)
with open("models/diabetes_features.pkl", "wb") as f: pickle.dump(list(X_d.columns), f)
with open("models/heart_features.pkl",    "wb") as f: pickle.dump(list(X_h.columns), f)
with open("models/metrics.pkl",           "wb") as f:
    pickle.dump({"diabetes":{"acc":acc_d,"auc":auc_d},"heart":{"acc":acc_h,"auc":auc_h}}, f)

print("  ✓ Models saved to models/")
print("\n✅ All done! Run:  streamlit run app.py")
