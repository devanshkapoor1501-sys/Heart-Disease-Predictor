
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("heart.csv")

df = pd.DataFrame(data)
df = df.drop_duplicates()
df["age"] = pd.to_numeric(df["age"])

x = df.drop(["restecg", "fbs", "chol", "target"], axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(
    C=0.1,
    max_iter=5000
)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
print("F1 Score:", f1)

joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")
st.write("Enter the patient's information below.")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex_option = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

sex_mapping = {
    "Female": 0,
    "Male": 1
}

sex = sex_mapping[sex_option]

cp_option = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_mapping[cp_option]

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang_option = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang_mapping = {
    "No": 0,
    "Yes": 1
}

exang = exang_mapping[exang_option]

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope_option = st.selectbox(
    "Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope = slope_mapping[slope_option]

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal_option = st.selectbox(
    "Thal",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

thal_mapping = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_mapping[thal_option]

if st.button("Predict"):

    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)

    if prediction[0] == 0:
        st.success("No heart disease detected.")
    else:
        st.error("Heart disease detected. Please consult a doctor.")

