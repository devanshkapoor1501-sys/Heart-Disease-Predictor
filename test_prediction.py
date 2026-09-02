import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from model_utils import (
    DATA_PATH,
    DATASET_TARGET_TO_APP_TARGET,
    FEATURE_COLUMNS,
    UI_MAPPINGS,
    build_input_data,
    load_dataset,
    predict_patient,
    train_model,
)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(DATA_PATH)
        cls.model = train_model(DATA_PATH, version=4)

    @staticmethod
    def _streamlit_prediction(values):
        app_path = Path(__file__).with_name("main.py.py")
        app = AppTest.from_file(str(app_path)).run(timeout=30)

        number_inputs = app.number_input
        selectboxes = app.selectbox
        number_inputs[0].set_value(int(values["age"]))
        selectboxes[0].set_value(
            {value: label for label, value in UI_MAPPINGS["sex"].items()}[
                int(values["sex"])
            ]
        )
        selectboxes[1].set_value(
            {value: label for label, value in UI_MAPPINGS["cp"].items()}[
                int(values["cp"])
            ]
        )
        number_inputs[1].set_value(int(values["trestbps"]))
        number_inputs[2].set_value(int(values["thalach"]))
        selectboxes[2].set_value(
            {value: label for label, value in UI_MAPPINGS["exang"].items()}[
                int(values["exang"])
            ]
        )
        number_inputs[3].set_value(float(values["oldpeak"]))
        selectboxes[3].set_value(
            {value: label for label, value in UI_MAPPINGS["slope"].items()}[
                int(values["slope"])
            ]
        )
        selectboxes[4].set_value(int(values["ca"]))
        selectboxes[5].set_value(
            {value: label for label, value in UI_MAPPINGS["thal"].items()}[
                int(values["thal"])
            ]
        )
        app.button[0].click().run(timeout=30)

        if app.error:
            return "disease"
        if app.success:
            return "no_disease"
        raise AssertionError("Streamlit produced no prediction message")

    def test_ui_mappings_use_training_values(self):
        for column, mapping in UI_MAPPINGS.items():
            available_values = set(self.data[column].astype(int).unique())
            self.assertTrue(set(mapping.values()) <= available_values)

    def test_input_data_preserves_training_feature_order(self):
        values = {column: 0 for column in FEATURE_COLUMNS}
        self.assertEqual(
            list(build_input_data(values).columns),
            FEATURE_COLUMNS,
        )

    def test_model_predicts_both_classes_on_held_out_data(self):
        X_train, X_test, y_train, _ = train_test_split(
            self.data[FEATURE_COLUMNS],
            self.data["target"].astype(int).map(DATASET_TARGET_TO_APP_TARGET),
            test_size=0.20,
            random_state=42,
            stratify=self.data["target"].astype(int).map(
                DATASET_TARGET_TO_APP_TARGET
            ),
        )
        del X_train, y_train
        predictions = set(self.model.predict(X_test).astype(int).tolist())
        self.assertEqual(predictions, {0, 1})

    def test_known_disease_and_no_disease_rows(self):
        disease_row = self.data[self.data["target"] == 0].iloc[0]
        no_disease_row = self.data[self.data["target"] == 1].iloc[0]

        disease_values = disease_row[FEATURE_COLUMNS].to_dict()
        no_disease_values = no_disease_row[FEATURE_COLUMNS].to_dict()

        self.assertEqual(predict_patient(self.model, disease_values), 1)
        self.assertEqual(predict_patient(self.model, no_disease_values), 0)

    def test_streamlit_button_displays_correct_result_for_both_classes(self):
        disease_row = self.data[self.data["target"] == 0].iloc[0]
        no_disease_row = self.data[self.data["target"] == 1].iloc[0]

        self.assertEqual(
            self._streamlit_prediction(disease_row[FEATURE_COLUMNS].to_dict()),
            "disease",
        )
        self.assertEqual(
            self._streamlit_prediction(
                no_disease_row[FEATURE_COLUMNS].to_dict()
            ),
            "no_disease",
        )


if __name__ == "__main__":
    unittest.main()
