import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df["flag_invoice"] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [100, 50, 20, 5],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [50.0, 20.0, 10.0, 5.0],
        "days_po_to_invoice": [5, 3, 2, 1],
        "days_to_pay": [30, 30, 15, 10],
        "total_brands": [10, 5, 2, 1],
        "total_item_quantity": [95, 48, 18, 5],
        "total_item_dollars": [18000, 8800, 2900, 190],
        "avg_receiving_delay": [2.0, 1.5, 0.0, 0.0]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)