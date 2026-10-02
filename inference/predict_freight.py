from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_freight_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    return model, scaler


def predict_freight_cost(input_data):
    model, scaler = load_artifacts()
    input_df = pd.DataFrame(input_data)

    if hasattr(scaler, "feature_names_in_"):
        input_df = input_df[scaler.feature_names_in_]

    scaled_data = scaler.transform(input_df)

    input_df['Predicted_Freight'] = model.predict(scaled_data).round()
    return input_df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    sample_data = {
        "invoice_quantity": [100, 50, 20, 5],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "days_po_to_invoice": [5, 3, 2, 1],
        "days_to_pay": [30, 30, 15, 10],
        "total_brands": [10, 5, 2, 1],
        "total_item_quantity": [95, 48, 18, 5],
        "total_item_dollars": [18000, 8800, 2900, 190],
        "avg_receiving_delay": [2.0, 1.5, 0.0, 0.0]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)