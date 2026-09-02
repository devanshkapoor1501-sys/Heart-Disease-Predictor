import streamlit as st

from model_utils import DATA_PATH, UI_MAPPINGS, predict_patient, train_model


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)


@st.cache_resource
def get_model():
    return train_model(DATA_PATH, version=4)


try:

    model = get_model()

except FileNotFoundError:

    st.error(
        f"heart.csv was not found at {DATA_PATH}."
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


sex_options = UI_MAPPINGS["sex"]

sex_text = st.selectbox(
    "Sex",
    list(sex_options.keys())
)

sex = sex_options[sex_text]


cp_options = UI_MAPPINGS["cp"]

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


exang_options = UI_MAPPINGS["exang"]

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


slope_options = UI_MAPPINGS["slope"]

slope_text = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    list(slope_options.keys())
)

slope = slope_options[slope_text]


ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)


thal_options = UI_MAPPINGS["thal"]

thal_text = st.selectbox(
    "Thalassemia Result",
    list(thal_options.keys())
)

thal = thal_options[thal_text]


input_values = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}


if st.button(
    "Predict Heart Disease",
    use_container_width=True
):

    try:

        prediction = predict_patient(model, input_values)

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
