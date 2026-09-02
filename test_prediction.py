import unittest

from sklearn.model_selection import train_test_split

from model_utils import (
    DATA_PATH,
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
        cls.model = train_model(DATA_PATH, version=3)

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
            self.data["target"].astype(int),
            test_size=0.20,
            random_state=42,
            stratify=self.data["target"].astype(int),
        )
        del X_train, y_train
        predictions = set(self.model.predict(X_test).astype(int).tolist())
        self.assertEqual(predictions, {0, 1})

    def test_known_positive_and_negative_rows(self):
        positive_row = self.data[self.data["target"] == 1].iloc[0]
        negative_row = self.data[self.data["target"] == 0].iloc[0]

        positive_values = positive_row[FEATURE_COLUMNS].to_dict()
        negative_values = negative_row[FEATURE_COLUMNS].to_dict()

        self.assertEqual(predict_patient(self.model, positive_values), 1)
        self.assertEqual(predict_patient(self.model, negative_values), 0)


if __name__ == "__main__":
    unittest.main()
