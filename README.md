# 🏥 HealthAI Dashboard

A Machine Learning-powered Healthcare Analytics Dashboard built using **Python**, **Streamlit**, **Scikit-learn**, and **Plotly**. The application provides an interactive platform for hospital analytics, patient insights, and disease risk prediction using trained machine learning models. It combines healthcare data visualization with predictive analytics to support better decision-making and demonstrate the practical application of AI in healthcare.

---

## 🚀 Features

### 📊 Hospital Overview Dashboard
- Monitor key hospital KPIs
- Total Patients
- Total Admissions
- Average Treatment Cost
- Average Length of Stay
- Department-wise Patient Distribution
- Patient Status Overview
- Monthly Admission Trends
- Age Distribution Analysis

### ❤️ Heart Disease Prediction
- Predict heart disease risk using a trained **Gradient Boosting Classifier**
- Interactive input form for patient medical information
- Risk probability prediction
- Feature Importance Visualization
- Model Confidence Score

### 🩸 Diabetes Prediction
- Predict diabetes risk using a **Random Forest Classifier**
- User-friendly medical input interface
- Probability-based prediction
- Feature Importance Analysis
- Interactive Result Dashboard

### 🏥 Patient Analytics
- Filter patient records dynamically
- Department-wise Cost Analysis
- Patient Satisfaction Analysis
- Length of Stay (LOS) Analysis
- Interactive Data Tables

### 📈 Model Performance
- Accuracy Comparison
- ROC-AUC Score
- Feature Importance Charts
- Distribution Analysis
- Model Evaluation Metrics

---

# 🤖 Machine Learning Models

## ❤️ Heart Disease Prediction

| Property | Value |
|----------|-------|
| Model | Gradient Boosting Classifier |
| Estimators | 100 |
| Accuracy | ~84% |
| ROC-AUC | ~0.91 |

### Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Major Vessels
- Thalassemia

---

## 🩸 Diabetes Prediction

| Property | Value |
|----------|-------|
| Model | Random Forest Classifier |
| Trees | 100 |
| Accuracy | ~85% |
| ROC-AUC | ~0.92 |

### Features Used

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

# 🛠️ Tech Stack

### Frontend
- Streamlit
- Plotly

### Backend
- Python

### Machine Learning
- Scikit-learn
- Random Forest
- Gradient Boosting

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly
- Streamlit Charts

---

# 📁 Project Structure

```text
healthcare_dashboard/
│
├── app.py
├── generate_data.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── patients.csv
│
└── models/
    ├── diabetes_model.pkl
    ├── heart_model.pkl
    ├── metrics.pkl
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/your-username/HealthAI-Dashboard.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Generate Dataset and Train Models

```bash
python generate_data.py
```

## Run Application

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

# 📸 Dashboard Preview

## 📊 Hospital Overview Dashboard

![Hospital Overview](./AI%20Health%20Analytics%20Dashboard01.png)

---

## ❤️ Heart Disease Prediction

![Heart Disease Prediction](./AI%20Health%20Analytics%20Dashboard02.png)

---

## 🩸 Diabetes Prediction

![Diabetes Prediction](./AI%20Health%20Analytics%20Dashboard03.png)

---

## 🏥 Patient Analytics

![Patient Analytics](./AI%20Health%20Analytics%20Dashboard04.png)

---

## 📈 Model Performance Dashboard

![Model Performance](./AI%20Health%20Analytics%20Dashboard05.png)

---

# ⭐ Key Highlights

- End-to-End Machine Learning Project
- Healthcare Analytics Dashboard
- Disease Risk Prediction
- Interactive Data Visualization
- Explainable AI using Feature Importance
- Hospital KPI Monitoring
- Patient Cost & Satisfaction Analysis
- Professional Streamlit UI
- Real-time Prediction Interface
- Data-driven Decision Support

---

# 👨‍💻 Author

**Tushar Kunwar**

### Connect with me

- LinkedIn: https://linkedin.com/in/your-profile
- GitHub: https://github.com/your-username

---

⭐ **If you found this project useful, consider giving it a Star on GitHub!**
