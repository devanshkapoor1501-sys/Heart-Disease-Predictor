import streamlit as st
import pandas as pd

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
def train_model(version=2):

    df = pd.read_csv("heart.csv")

    df = df.drop_duplicates()

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
        (
            "scaler",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=5000
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    return model


try:

    model = train_model()

except FileNotFoundError:

    st.error(
        "heart.csv was not found. Make sure it is in the same folder as app.py."
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


sex_options = {
    "Female": 0,
    "Male": 1
}

sex_text = st.selectbox(
    "Sex",
    list(sex_options.keys())
)

sex = sex_options[sex_text]


cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp_text = st.selectbox(
    "Chest Pain Type",
    list(cp_options.keys())
)

cp = cp_options[cp_text]


trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=220,
    value=120,
    step=1
)


thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=70,
    max_value=210,
    value=150,
    step=1
)


exang_options = {
    "No": 0,
    "Yes": 1
}

exang_text = st.selectbox(
    "Exercise Induced Angina",
    list(exang_options.keys())
)

exang = exang_options[exang_text]


oldpeak = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=6.5,
    value=1.0,
    step=0.1
)


slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope_text = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    list(slope_options.keys())
)

slope = slope_options[slope_text]


ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)


thal_options = {
    "Unknown / No Result": 0,
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal_text = st.selectbox(
    "Thalassemia Result",
    list(thal_options.keys())
)

thal = thal_options[thal_text]


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
