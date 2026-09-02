import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


@st.cache_resource
def train_model():

    df = pd.read_csv("heart.csv")

    df = df.drop_duplicates()

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    X = df[FEATURE_COLUMNS].copy()

    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000))
    ])

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        "heart_model.pkl"
    )

    return model


try:

    model = train_model()

except FileNotFoundError:

    st.error(
        "heart.csv was not found."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error training model: {e}"
    )

    st.stop()


st.title("❤️ Heart Disease Predictor")

st.write(
    "Enter the patient's information below."
)


st.subheader("Patient Information")


age = st.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=50,
    step=1
)


sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x:
    "Female" if x == 0 else "Male"
)


cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3],
    format_func=lambda x:
    {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }[x]
)


trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=120,
    step=1
)


thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=70,
    max_value=210,
    value=150,
    step=1
)


exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1],
    format_func=lambda x:
    "No" if x == 0 else "Yes"
)


oldpeak = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=6.5,
    value=1.0,
    step=0.1
)


slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)


ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)


thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)


input_data = pd.DataFrame(
    [[
        age,
        sex,
        cp,
        trestbps,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]],
    columns=FEATURE_COLUMNS
)


if st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
):

    try:

        prediction = model.predict(
            input_data
        )[0]

        if prediction == 1:

            st.error(
                "Heart disease detected."
            )

        else:

            st.success(
                "No heart disease detected."
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
