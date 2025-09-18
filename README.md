# California Housing Prices – Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains an end-to-end machine learning project for predicting California housing prices using the famous dataset from the 1990 California census. The project includes comprehensive exploratory data analysis, feature engineering, model development, and a production-ready web application.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Web Application](#web-application)
- [Key Insights](#key-insights)
- [Future Work](#future-work)
- [References](#references)

## 🎯 Project Overview

This project aims to build accurate predictive models for estimating median house values in California districts. The implementation follows a complete machine learning pipeline:

- Comprehensive exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and selection
- Model training and evaluation
- Deployment via web application

The project is implemented in **Jupyter Notebook** and developed using **GitHub Codespaces**.

## 📊 Dataset

**Source:** Originally from R. Kelley Pace and Ronald Barry (1997), modified by Aurélien Géron for "Hands-On Machine Learning with Scikit-Learn and TensorFlow".

**Features:**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of houses in the district
- `total_rooms` - Total number of rooms in the district
- `total_bedrooms` - Total number of bedrooms in the district
- `population` - Population of the district
- `households` - Number of households in the district
- `median_income` - Median income of households (scaled)
- `ocean_proximity` - Categorical variable indicating proximity to ocean
- `median_house_value` - Target variable (median house value in USD)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/agik20/californiahousinprice.git
cd californiahousinprice
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
californiahousinprice/
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks for EDA and modeling
├── src/                     # Source code for data processing and modeling
├── models/                  # Trained model files
├── app/                     # Flask web application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Containerization configuration
└── README.md
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of all features
- Correlation analysis between variables
- Geospatial visualization of housing prices
- Identification of data quality issues

### 2. Data Preprocessing
- Handling missing values in `total_bedrooms` column
- One-hot encoding of categorical variable `ocean_proximity`
- Standardization of numerical features

### 3. Feature Engineering
Created more informative derived features:
- `rooms_per_household` - Average number of rooms per household
- `bedrooms_per_room` - Proportion of bedrooms to total rooms
- `population_per_household` - Average household size

### 4. Modeling
Implemented and evaluated multiple algorithms:
- **Linear Regression** (baseline model)
- **Random Forest Regressor** (primary model)
- Evaluation using Root Mean Squared Error (RMSE)

### 5. Model Interpretation
- Analysis of feature importance
- Model performance comparison
- Error analysis and visualization

## 📈 Results

### Variable Distribution
![Histogram for each feature](https://github.com/user-attachments/assets/b83a1295-3691-4490-a6ac-16d5679310bb)

### Feature Correlation Matrix
![Feature and target correlation](https://github.com/user-attachments/assets/f4272a2d-2f5b-404c-9a13-8b7975449e02)

### Engineered Features Distribution
![Histogram for each feature (feature engineering)](https://github.com/user-attachments/assets/dcd57bb7-d68d-4a3d-88e0-3ea3fef31d11)

### One-Hot Encoding Correlation Heatmap
![One-hot encoding result correlation heatmap](https://github.com/user-attachments/assets/9cd68535-6fe9-4871-8d60-d56b57f010a7)

## 🌐 Web Application

The project includes a Flask web application for making predictions interactively.

### Run Locally
```bash
python app.py
```
Then open: http://localhost:5000

### Run with Docker
```bash
docker build -t house-price-app .
docker run -d -p 80:5000 house-price-app
```
Then access at: http://localhost

## 💡 Key Insights

1. **Median income** is the most influential factor in predicting house prices (correlation ≈ 0.69)
2. **Geographic location** significantly affects prices:
   - Coastal properties (<1H Ocean / Near Bay) command premium prices
   - Inland properties are generally more affordable
3. **Feature engineering** substantially improved model performance:
   - Ratio features (rooms per household, etc.) provided more meaningful signals
   - Reduced multicollinearity among features
4. **Tree-based models** (Random Forest) outperformed linear models by capturing non-linear relationships

## 🚀 Future Work

- Experiment with advanced boosting algorithms (XGBoost, LightGBM, CatBoost)
- Implement systematic hyperparameter optimization (GridSearchCV/RandomizedSearchCV)
- Develop interactive geospatial visualizations with Folium or GeoPandas
- Create automated ML pipeline with MLflow for experiment tracking
- Implement model monitoring and drift detection in production

## 📚 References

1. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn and TensorFlow. O'Reilly Media.
2. Pace, R. K., & Barry, R. (1997). Quick Computation of Regression with a Spatial Autoregressive Weight. Computational Statistics & Data Analysis.
3. Glaeser, E. L., & Gyourko, J. (2008). Rethinking Federal Housing Policy. American Enterprise Institute.

---

**Disclaimer**: This project is intended for educational purposes as an exercise in end-to-end machine learning pipeline development.

---
*📧 For questions or suggestions, please open an issue or contact the repository maintainer.*
