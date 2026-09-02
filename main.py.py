import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    param_grid = {
        "model__C": [
            0.001,
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cv_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    cv_accuracy = cv_scores.mean()

    joblib.dump(
        best_model,
        "heart_model.pkl"
    )

    return (
        best_model,
        accuracy,
        cv_accuracy,
        grid.best_params_,
        df
    )


@st.cache_resource
def load_saved_model():

    try:
        return joblib.load(
            "heart_model.pkl"
        )

    except:
        return None


st.title("❤️ Heart Disease Predictor")

st.write(
    "Enter the patient's information below."
)


try:

    saved_model = load_saved_model()

    if saved_model is None:

        (
            model,
            accuracy,
            cv_accuracy,
            best_params,
            df
        ) = train_model()

    else:

        model = saved_model

        df = pd.read_csv(
            "heart.csv"
        )

        df = df.drop_duplicates()

        df["age"] = pd.to_numeric(
            df["age"],
            errors="coerce"
        )

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

        y_test_pred = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_test_pred
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        cv_accuracy = cv_scores.mean()

        best_params = {
            "C": model.named_steps["model"].C
        }


except FileNotFoundError:

    st.error(
        "heart.csv was not found. Put heart.csv in the same folder as app.py."
    )

    st.stop()


except Exception as e:

    st.error(
        f"Error loading the model or dataset: {e}"
    )

    st.stop()


st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

st.sidebar.write(
    f"Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%"
)

st.sidebar.write(
    f"Best C: {best_params['C']}"
)


st.subheader(
    "Patient Information"
)


age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
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
    [0, 1, 2, 3]
)


trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)


thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
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
    max_value=10.0,
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


st.subheader(
    "Input Data"
)

st.dataframe(
    input_data,
    use_container_width=True
)


if st.button(
    "Predict Heart Disease",
    use_container_width=True
):

    try:

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        class_0_probability = probabilities[0]

        class_1_probability = probabilities[1]


        st.subheader(
            "Prediction Result"
        )


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
            f"{class_0_probability * 100:.2f}%"
        )

        st.write(
            f"Probability of class 1: "
            f"{class_1_probability * 100:.2f}%"
        )


        st.write(
            "Class 0"
        )

        st.progress(
            float(class_0_probability)
        )


        st.write(
            "Class 1"
        )

        st.progress(
            float(class_1_probability)
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
