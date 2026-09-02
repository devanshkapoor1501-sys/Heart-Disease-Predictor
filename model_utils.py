from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

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
    "thal",
]


UI_MAPPINGS = {
    "sex": {
        "Female": 0,
        "Male": 1,
    },
    "cp": {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3,
    },
    "exang": {
        "No": 0,
        "Yes": 1,
    },
    "slope": {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2,
    },
    "thal": {
        "Unknown / No Result": 0,
        "Normal": 3,
        "Fixed Defect": 1,
        "Reversible Defect": 2,
    },
}


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the dataset used by both training and inference."""
    df = pd.read_csv(data_path).drop_duplicates().copy()
    required_columns = FEATURE_COLUMNS + ["target"]
    missing_columns = [column for column in required_columns if column not in df]

    if missing_columns:
        raise ValueError(
            "heart.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df[required_columns].isna().any().any():
        raise ValueError("heart.csv contains missing values in model columns")

    target_values = set(df["target"].astype(int).unique())
    if target_values != {0, 1}:
        raise ValueError(
            "heart.csv target must contain both 0 (no disease) and "
            f"1 (disease); found {sorted(target_values)}"
        )

    return df


def train_model(data_path: Path = DATA_PATH, version: int = 3) -> Pipeline:
    """Train the serving model and reject a dataset/model that collapses to one class."""
    del version  # Explicit cache-busting argument for Streamlit callers.
    df = load_dataset(data_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].astype(int)

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(X_train, y_train)

    held_out_predictions = set(model.predict(X_test).astype(int).tolist())
    if held_out_predictions != {0, 1}:
        raise ValueError(
            "Model sanity check failed: held-out predictions contained "
            f"only {sorted(held_out_predictions)}"
        )

    return model


def build_input_data(values: Mapping[str, object]) -> pd.DataFrame:
    """Build one prediction row in exactly the order used during training."""
    missing_values = [column for column in FEATURE_COLUMNS if column not in values]
    if missing_values:
        raise ValueError(
            "Missing prediction values: " + ", ".join(missing_values)
        )

    return pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def predict_patient(model: Pipeline, values: Mapping[str, object]) -> int:
    """Return a validated binary prediction for one patient."""
    prediction_values = model.predict(build_input_data(values))
    if len(prediction_values) != 1:
        raise ValueError("Expected exactly one prediction")

    prediction = int(prediction_values[0])
    if prediction not in (0, 1):
        raise ValueError(f"Unexpected model prediction: {prediction}")

    return prediction
