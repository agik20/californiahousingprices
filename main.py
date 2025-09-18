import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PLOT_DIR = "plots"


def save_plot(name):
    """Helper untuk save plot ke folder plots"""
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    plt.savefig(os.path.join(PLOT_DIR, name))
    plt.close()


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Print evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {model_name} Evaluation ===")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


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
    train_data = X_train.join(y_train)

    print("[3] Exploratory plots...")
    train_data.hist(figsize=(15, 8))
    save_plot("train_hist_before.png")

    plt.figure(figsize=(15, 8))
    sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    save_plot("train_corr_before.png")

    print("[4] Feature engineering (log transform + ratios)...")
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        train_data[col] = np.log(train_data[col] + 1)

    train_data.hist(figsize=(15, 8))
    save_plot("train_hist_after.png")

    train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(
        ["ocean_proximity"], axis=1
    )

    plt.figure(figsize=(15, 8))
    sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    save_plot("train_corr_after.png")

    plt.figure(figsize=(15, 8))
    sns.scatterplot(
        x="latitude",
        y="longitude",
        data=train_data,
        hue="median_house_value",
        palette="coolwarm",
    )
    save_plot("train_scatter.png")

    train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
    train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

    plt.figure(figsize=(15, 8))
    sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
    save_plot("train_corr_final.png")

    print("[5] Scaling data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_data.drop(["median_house_value"], axis=1))
    y_train = train_data["median_house_value"]

    print("[6] Train Linear Regression...")
    reg = LinearRegression()
    reg.fit(X_train_s, y_train)

    print("[7] Prepare test set...")
    test_data = X_test.join(y_test)
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        test_data[col] = np.log(test_data[col] + 1)
    test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(
        ["ocean_proximity"], axis=1
    )
    test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
    test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

    X_test_s = scaler.transform(test_data.drop(["median_house_value"], axis=1))
    y_test = test_data["median_house_value"]

    print("[8] Evaluate Linear Regression...")
    y_pred_lr = reg.predict(X_test_s)
    evaluate_model(y_test, y_pred_lr, "Linear Regression")

    print("[9] Train Random Forest + GridSearch...")
    forest = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=forest,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_s, y_train)
    best_forest = grid_search.best_estimator_

    print("Best params:", grid_search.best_params_)
    print("Best score (CV):", grid_search.best_score_)

    print("[10] Evaluate Random Forest...")
    y_pred_rf = best_forest.predict(X_test_s)
    evaluate_model(y_test, y_pred_rf, "Random Forest")


if __name__ == "__main__":
    main()
