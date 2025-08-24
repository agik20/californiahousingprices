# California Housing Prices – Machine Learning Project

This repository contains an exploratory data analysis (EDA) and machine learning implementation on the **California Housing Prices dataset**.  
The dataset originates from the 1990 California census and is famously used in *Hands-On Machine Learning with Scikit-Learn and TensorFlow* by Aurélien Géron.

---

## 📌 Project Overview
The goal of this project is to:
- Explore the California Housing dataset.
- Perform data cleaning and preprocessing.
- Visualize key patterns and correlations.
- Train predictive models to estimate **median_house_value**.

This project is implemented in **Jupyter Notebook (`.ipynb`)** and developed using **GitHub Codespaces**.

---

## 📌 Problem Statement
Housing is a crucial aspect of social and economic life, as house prices influence people's ability to afford adequate housing and impact urban development and economic policy (Glaeser & Gyourko, 2008). With population growth and urbanization, accurate house price predictions are crucial for developers, buyers, and policymakers to make informed decisions.

Several previous studies have shown that house prices are influenced by various factors, including property characteristics (number of bedrooms, house size), economic conditions (median income), demographics, and geographic factors such as proximity to public amenities or beaches (Pace & Barry, 1997). For example, Pace and Barry (1997) developed a spatial autoregressive model to predict house prices in California, emphasizing the importance of spatial relationships between districts. Other studies highlight the importance of modern machine learning techniques, such as Random Forest and Gradient Boosting, in improving the accuracy of house price predictions compared to classical linear regression models (A. Géron, 2022).

Although the California Housing Prices dataset has been widely used for machine learning, there is still a need to understand the combined influence of numerical and geographic features on house prices and how preprocessing and feature engineering can improve model performance. This research aims to fill this gap by building a comprehensive predictive pipeline that combines data exploration, feature engineering, and modern machine learning models.

---

## 📂 Dataset
- **Source:** Originally from R. Kelley Pace and Ronald Barry (1997), modified by Aurélien Géron.  
- **Columns:**
  - `longitude`  
  - `latitude`  
  - `housing_median_age`  
  - `total_rooms`  
  - `total_bedrooms`  
  - `population`  
  - `households`  
  - `median_income`  
  - `median_house_value`
  - `ocean_proximity`
- **Target:** 'median_house_value'
---

## ⚙️ Requirements
To run this project locally or in a codespace, install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔎 Langkah-Langkah Analisis / Pipeline
1. **Loading Data**
Importing the California Housing dataset.

2. **Exploratory Data Analysis (EDA)**
- Histogram of the distribution of each feature.
- Heatmap of correlations between features.
- Geospatial analysis (longitude vs. latitude).

3. **Data Cleaning & Preprocessing**
- Handling missing values ​​(`total_bedrooms`).
- Categorical encoding (`ocean_proximity` → one-hot).
- Scaling/normalizing numeric variables.

4. **Feature Engineering**
Creating more informative derived features:
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

5. **Modeling**
- Baseline: **Linear Regression**
- Tree-based model: **Random Forest Regressor**
- Evaluation using RMSE (Root Mean Squared Error).

6. **Evaluation & Comparison**
- Compare baseline results vs. complex models.
- Feature importance interpretation.

## 📈 Output Example / Visualization
### Variable Distribution
<img width="1247" height="682" alt="histogram for each feature" src="https://github.com/user-attachments/assets/b83a1295-3691-4490-a6ac-16d5679310bb" />

### Correlation Between Features
<img width="1247" height="797" alt="feature and target correlation" src="https://github.com/user-attachments/assets/f4272a2d-2f5b-404c-9a13-8b7975449e02" />

### Histogram after Feature Engineering
<img width="1238" height="682" alt="histogram for each feature (feature engineering)" src="https://github.com/user-attachments/assets/dcd57bb7-d68d-4a3d-88e0-3ea3fef31d11" />

### One-Hot Encoding Result Correlation Heatmap
<img width="1247" height="797" alt="one-hot encoding result correlation heatmap" src="https://github.com/user-attachments/assets/9cd68535-6fe9-4871-8d60-d56b57f010a7" />


## 💡 Key Insights / Conclusions
- Median income is the most dominant factor influencing house prices (correlation ≈ 0.69).
- Location also plays a role: houses inland tend to be cheaper, while houses <1H Ocean / Near Bay are more expensive.
- Raw variables such as total rooms, population, and households are highly correlated with each other, making them better represented in ratio form.
- A simple model (Linear Regression) already provides a baseline, but a tree-based model (Random Forest) is able to capture non-linearity and provide more accurate predictions.

---

## 🚀 Next Steps
- Trying model boosting (XGBoost, LightGBM, CatBoost).
- More systematic hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
- Interactive geospatial visualization with `folium` or `geopandas`.

---

📌 *This project is intended as an exercise in the end-to-end machine learning pipeline, from data exploration to modeling, with a focus on interpretation and feature engineering.*
