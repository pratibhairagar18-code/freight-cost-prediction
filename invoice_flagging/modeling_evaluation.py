import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def load_invoice_data(db_path='data/inventory.db'):
    conn = sqlite3.connect(db_path)
    
    query = """
    WITH purchase_agg AS (
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT 
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def split_data(df, features, target):
    # Drop rows where Freight or input features might have missing values
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def scale_features(X_train, X_test, scaler_path='models/scaler.pkl'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Preserve feature names in scaler for inference validation
    scaler.feature_names_in_ = X_train.columns.tolist()
    
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, scaler.feature_names_in_


def train_random_forest(X_train, y_train, model_path='models/predict_freight_model.pkl'):
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_path)
    print(f"\nModel saved successfully to {model_path}")

    return best_model


def evaluate_regressor(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\n{model_name} Performance")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")


# Execution Pipeline
if __name__ == "__main__":
    # 1. Load data
    df = load_invoice_data()

    # 2. Define features and continuous target variable (Freight)
    features = [
        'invoice_quantity', 'invoice_dollars', 
        'days_po_to_invoice', 'days_to_pay', 'total_brands', 
        'total_item_quantity', 'total_item_dollars', 'avg_receiving_delay'
    ]
    target = 'Freight'

    # 3. Train-test split & Scaling
    X_train, X_test, y_train, y_test = split_data(df, features, target)
    X_train_scaled, X_test_scaled, feature_names = scale_features(X_train, X_test)

    # 4. Train Model via Grid Search
    best_model = train_random_forest(X_train_scaled, y_train)

    # 5. Evaluate Performance
    evaluate_regressor(best_model, X_test_scaled, y_test, 'Random Forest Regressor')