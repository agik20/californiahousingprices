# housing_xgboost.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

PLOT_DIR = "xgboost-plots"

def save_plot(name):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    plt.savefig(os.path.join(PLOT_DIR, name))
    plt.close()

def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {model_name} Evaluation ===")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Predicted vs Actual")
    save_plot(f"{model_name}_pred_vs_actual.png")

def preprocess_data(df):
    df = df.copy()
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        df[col] = np.log(df[col] + 1)

    df = df.join(pd.get_dummies(df.ocean_proximity)).drop(["ocean_proximity"], axis=1)

    df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
    df["household_rooms"] = df["total_rooms"] / df["households"]

    return df

def main():
    print("[1] Load dataset...")
    data = pd.read_csv("housing.csv")
    data.dropna(inplace=True)

    print("[2] Split train/test...")
    X = data.drop(["median_house_value"], axis=1)
    y = data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[3] Preprocessing...")
    train_data = preprocess_data(X_train.join(y_train))
    test_data = preprocess_data(X_test.join(y_test))

    print("[4] Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_data.drop(["median_house_value"], axis=1))
    y_train = train_data["median_house_value"]
    X_test_s = scaler.transform(test_data.drop(["median_house_value"], axis=1))
    y_test = test_data["median_house_value"]

    print("[5] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train_s, y_train)

    print("[6] Evaluating XGBoost...")
    y_pred = xgb_model.predict(X_test_s)
    evaluate_model(y_test, y_pred, model_name="XGBoost")

if __name__ == "__main__":
    main()
