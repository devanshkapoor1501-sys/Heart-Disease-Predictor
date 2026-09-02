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

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    joblib.dump(
        pipeline,
        "heart_model.pkl"
    )

    return pipeline


try:

    model = train_model()

except FileNotFoundError:

    st.error(
        "heart.csv was not found. Put heart.csv in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error loading dataset or training model: {e}"
    )

    st.stop()


st.title("❤️ Heart Disease Predictor")

st.write(
    "Enter the patient's information below to predict the possibility of heart disease."
)


st.subheader("Patient Information")


age = st.number_input(
    "What is the patient's age?",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)


sex_option = st.selectbox(
    "What is the patient's sex?",
    [
        "Female",
        "Male"
    ]
)

if sex_option == "Female":
    sex = 0
else:
    sex = 1


cp_option = st.selectbox(
    "What type of chest pain does the patient have?",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

cp_values = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_values[cp_option]


trestbps = st.number_input(
    "What is the resting blood pressure (mm Hg)?",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)


thalach = st.number_input(
    "What is the maximum heart rate achieved?",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)


exang_option = st.selectbox(
    "Does exercise cause chest pain (angina)?",
    [
        "No",
        "Yes"
    ]
)

if exang_option == "No":
    exang = 0
else:
    exang = 1


oldpeak = st.number_input(
    "What is the ST depression caused by exercise?",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


slope_option = st.selectbox(
    "What is the slope of the peak exercise ST segment?",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

slope_values = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

slope = slope_values[slope_option]


ca = st.number_input(
    "Number of major vessels colored by fluoroscopy?",
    min_value=0,
    max_value=4,
    value=0,
    step=1
)


thal_option = st.selectbox(
    "What is the thalassemia result?",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

thal_values = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

thal = thal_values[thal_option]


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


st.subheader("Patient Data")

st.dataframe(
    input_data,
    use_container_width=True
)


if st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
):

    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        probability_class_0 = probabilities[0]
        probability_class_1 = probabilities[1]


        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Heart disease detected."
            )

        else:

            st.success(
                "✅ No heart disease detected."
            )


        st.write(
            f"Prediction: `{prediction}`"
        )

        st.write(
            f"Probability of class 0: "
            f"{probability_class_0 * 100:.2f}%"
        )

        st.write(
            f"Probability of class 1: "
            f"{probability_class_1 * 100:.2f}%"
        )


        st.subheader("Prediction Probability")


        st.write(
            "Class 0"
        )

        st.progress(
            float(probability_class_0)
        )


        st.write(
            "Class 1"
        )

        st.progress(
            float(probability_class_1)
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
