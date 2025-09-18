# California Housing Prices – Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning project for predicting California housing prices using census data. This project demonstrates a complete end-to-end pipeline from exploratory data analysis to model deployment.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)
- [References](#-references)

## 🎯 Project Overview

This repository contains an exploratory data analysis (EDA) and machine learning implementation on the **California Housing Prices dataset**, originally from the 1990 California census. The project follows the methodology presented in *"Hands-On Machine Learning with Scikit-Learn and TensorFlow"* by Aurélien Géron.

**Objective**: Develop accurate predictive models for estimating median house values in California districts based on demographic, economic, and geographic features.

## 📊 Dataset

The dataset contains housing information from the 1990 California census with the following features:

**Numerical Features:**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of houses in the district
- `total_rooms` - Total number of rooms in the district
- `total_bedrooms` - Total number of bedrooms in the district
- `population` - Population in the district
- `households` - Number of households in the district
- `median_income` - Median income of households (in tens of thousands of USD)
- `median_house_value` - Target variable (in USD)

**Categorical Feature:**
- `ocean_proximity` - Proximity to the ocean (5 categories)

**Source**: R. Kelley Pace and Ronald Barry (1997), modified by Aurélien Géron.

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of all features
- Correlation analysis between features and target variable
- Geospatial visualization of housing prices across California

### 2. Data Preprocessing
- Handling missing values in `total_bedrooms` column
- One-hot encoding for categorical variable `ocean_proximity`
- Standardization of numerical features

### 3. Feature Engineering
Created more informative derived features:
- `rooms_per_household` = total_rooms / households
- `bedrooms_per_room` = total_bedrooms / total_rooms
- `population_per_household` = population / households

### 4. Modeling Approaches
- **Baseline Model**: Linear Regression
- **Advanced Model**: Random Forest Regressor
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

### 5. Model Deployment
- Web application built with Flask for real-time predictions
- Docker containerization for easy deployment

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Local Installation
```bash
# Clone the repository
git clone https://github.com/agik20/californiahousinprice.git
cd californiahousinprice

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation
```bash
# Build the Docker image
docker build -t house-price-app .

# Run the container
docker run -d -p 5000:5000 house-price-app
```

## 💻 Usage

### Running the Analysis
Open and run the Jupyter Notebook (`california_housing.ipynb`) to explore the complete analysis pipeline.

### Launching the Web Application
```bash
python app.py
```
Access the application at: http://localhost:5000

### Making Predictions via API
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, \
"total_rooms": 880, "total_bedrooms": 129, "population": 322, \
"households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}' \
http://localhost:5000/predict
```

## 📈 Results

### Visualizations

#### Feature Distributions
![Feature Distributions](https://github.com/user-attachments/assets/b83a1295-3691-4490-a6ac-16d5679310bb)

#### Feature Correlations
![Correlation Heatmap](https://github.com/user-attachments/assets/f4272a2d-2f5b-404c-9a13-8b7975449e02)

#### Engineered Features
![Engineered Features](https://github.com/user-attachments/assets/dcd57bb7-d68d-4a3d-88e0-3ea3fef31d11)

#### Encoded Features Correlation
![One-Hot Encoding Correlation](https://github.com/user-attachments/assets/9cd68535-6fe9-4871-8d60-d56b57f010a7)

### Model Performance
- **Linear Regression**: RMSE = $68,000-72,000
- **Random Forest Regressor**: RMSE = $48,000-52,000 (≈30% improvement)

## 🔍 Key Insights

1. **Income Dominance**: Median income shows the strongest correlation with house prices (≈0.69)
2. **Location Impact**: Coastal properties (<1H Ocean / Near Bay) command premium prices
3. **Feature Engineering Benefits**: Derived ratio features (rooms per household, etc.) provided more predictive power than raw counts
4. **Non-linear Relationships**: Tree-based models outperformed linear regression, indicating complex non-linear patterns in the data
5. **Spatial Patterns**: Clear geographic clustering of housing prices, with coastal areas and urban centers showing higher values

## 🔮 Future Work

- Experiment with advanced boosting algorithms (XGBoost, LightGBM, CatBoost)
- Implement systematic hyperparameter optimization (GridSearchCV/RandomizedSearchCV)
- Develop interactive geospatial visualizations using Folium or GeoPandas
- Incorporate additional data sources (crime rates, school quality, amenities)
- Implement deep learning approaches for potentially improved performance
- Develop a more sophisticated web interface with interactive visualizations

## 📚 References

1. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
2. Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. Statistics & Probability Letters, 33(3), 291-297.
3. Glaeser, E. L., & Gyourko, J. (2008). Rethinking federal housing policy. American Enterprise Institute.

---

**Note**: This project is intended as an educational demonstration of end-to-end machine learning pipeline implementation, with emphasis on interpretability and feature engineering.

For questions or contributions, please open an issue or submit a pull request.
