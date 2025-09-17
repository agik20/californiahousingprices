🏠 California Housing Prices – Machine Learning Project

https://img.shields.io/badge/Machine-Learning-ff6b6b?style=for-the-badge
https://img.shields.io/badge/Python-3.8%2B-3776ab?style=for-the-badge
https://img.shields.io/badge/Scikit--Learn-1.2%2B-f7931e?style=for-the-badge
https://img.shields.io/badge/Jupyter-Notebook-f37626?style=for-the-badge

This repository contains an exploratory data analysis (EDA) and machine learning implementation on the California Housing Prices dataset. The dataset originates from the 1990 California census and is famously used in Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron.

📊 Project Overview

The goal of this project is to:

· Explore the California Housing dataset through comprehensive analysis
· Perform data cleaning and preprocessing
· Visualize key patterns and correlations
· Train predictive models to estimate median_house_value
· Deploy a web application for real-time predictions

This project is implemented in Jupyter Notebook (`.ipynb`) and developed using GitHub Codespaces.

---

🎯 Research Problem

Housing is a crucial aspect of social and economic life, as house prices influence people's ability to afford adequate housing and impact urban development and economic policy (Glaeser & Gyourko, 2008). With population growth and urbanization, accurate house price predictions are crucial for developers, buyers, and policymakers to make informed decisions.

📚 Literature Review

Several previous studies have shown that house prices are influenced by various factors:

· Property characteristics (number of bedrooms, house size)
· Economic conditions (median income)
· Demographics and geographic factors (proximity to public amenities or beaches)

Pace and Barry (1997) developed a spatial autoregressive model to predict house prices in California, emphasizing the importance of spatial relationships between districts. Modern machine learning techniques, such as Random Forest and Gradient Boosting, have shown improved accuracy compared to classical linear regression models (A. Géron, 2022).

🎯 Research Gap

Although the California Housing Prices dataset has been widely used for machine learning, there is still a need to understand:

· The combined influence of numerical and geographic features on house prices
· How preprocessing and feature engineering can improve model performance
· The practical deployment of models for real-world prediction

This research aims to fill this gap by building a comprehensive predictive pipeline that combines data exploration, feature engineering, and modern machine learning models.

---

📁 Dataset Description

Feature Description Type
longitude Longitudinal coordinate Numerical
latitude Latitudinal coordinate Numerical
housing_median_age Median age of housing in block Numerical
total_rooms Total number of rooms in block Numerical
total_bedrooms Total number of bedrooms in block Numerical
population Total population in block Numerical
households Total number of households in block Numerical
median_income Median income of households Numerical
median_house_value Median house value (target) Numerical
ocean_proximity Proximity to ocean (categorical) Categorical

Target Variable: median_house_value

---

🛠️ Installation & Setup

Prerequisites

· Python 3.8+
· Git
· Jupyter Notebook/JupyterLab

1. Clone the Repository

```bash
git clone https://github.com/agik20/californiahousinprice.git
cd californiahousinprice
```

2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# or
venv\Scripts\activate     # Windows
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook

```bash
jupyter notebook
```

---

📈 Analysis Pipeline

1. Data Loading & Initial Exploration

· Importing the California Housing dataset
· Basic statistical summary
· Data structure examination

2. Exploratory Data Analysis (EDA)

· Distribution Analysis: Histograms for each feature
· Correlation Analysis: Heatmap of feature correlations
· Geospatial Analysis: Latitude vs. longitude visualization
· Target Analysis: Distribution of median house values

3. Data Cleaning & Preprocessing

· Handling missing values (total_bedrooms)
· Categorical encoding (ocean_proximity → one-hot encoding)
· Scaling/normalizing numeric variables
· Outlier detection and treatment

4. Feature Engineering

Creating more informative derived features:

· rooms_per_household = total_rooms / households
· bedrooms_per_room = total_bedrooms / total_rooms
· population_per_household = population / households

5. Modeling & Evaluation

· Baseline Model: Linear Regression
· Advanced Model: Random Forest Regressor
· Evaluation Metric: RMSE (Root Mean Squared Error)
· Cross-validation: K-fold validation for robustness

6. Model Interpretation

· Feature importance analysis
· Residual analysis
· Model comparison and selection

---

📊 Key Visualizations

Variable Distribution

https://github.com/user-attachments/assets/b83a1295-3691-4490-a6ac-16d5679310bb

Histograms showing the distribution of each feature in the dataset

Feature Correlation Heatmap

https://github.com/user-attachments/assets/f4272a2d-2f5b-404c-9a13-8b7975449e02

Correlation matrix showing relationships between features and the target variable

Engineered Features Distribution

https://github.com/user-attachments/assets/dcd57bb7-d68d-4a3d-88e0-3ea3fef31d11

Distribution of features after engineering, showing improved characteristics

One-Hot Encoding Correlation

https://github.com/user-attachments/assets/9cd68535-6fe9-4871-8d60-d56b57f010a7

Correlation heatmap after one-hot encoding of categorical features

---

🌐 Web Application Deployment

Option 1: Run Locally

```bash
git clone https://github.com/agik20/californiahousinprice.git
cd californiahousinprice
pip install -r requirements.txt
python app.py
```

Access the application at: 👉 http://localhost:5000

Option 2: Run with Docker

```bash
docker build -t house-price-app .
docker run -d -p 80:5000 house-price-app
```

Access the application at: 👉 http://localhost

Application Features

· Interactive input form for housing features
· Real-time price prediction
· Model information and performance metrics
· Responsive design for all devices

---

🔍 Key Insights

1. Strong Income Correlation: Median income is the most dominant factor influencing house prices (correlation ≈ 0.69)
2. Geographic Influence: Location plays a significant role - houses inland tend to be cheaper, while houses near the ocean (<1H Ocean / Near Bay) are more expensive
3. Feature Engineering Benefits: Raw variables (total rooms, population, households) are highly correlated and better represented as ratios
4. Model Performance:
   · Linear Regression provides a reasonable baseline
   · Random Forest captures non-linear relationships and provides more accurate predictions

---

🚀 Future Enhancements

Immediate Improvements

· Implement advanced boosting algorithms (XGBoost, LightGBM, CatBoost)
· Systematic hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
· Interactive geospatial visualization with folium or geopandas

Medium-Term Goals

· Time-series analysis for housing price trends
· Integration of additional data sources (crime rates, school quality)
· Advanced feature engineering with domain knowledge

Long-Term Vision

· Real-time API for housing price predictions
· Mobile application development
· Regional housing market comparison tools

---

📚 References

1. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
2. Glaeser, E. L., & Gyourko, J. (2008). Rethinking Federal Housing Policy. American Enterprise Institute.
3. Pace, R. K., & Barry, R. (1997). Quick Computation of Regression with a Spatially Autoregressive Dependent Variable. Geographical Analysis.

---

🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: git checkout -b feature/amazing-feature
3. Commit your changes: git commit -m 'Add amazing feature'
4. Push to the branch: git push origin feature/amazing-feature
5. Open a Pull Request

---

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

🙋‍♂️ Support

If you encounter any issues:

1. Check the existing GitHub Issues
2. Create a new issue with detailed information about your problem
3. Include your environment details and error messages

---

<div align="center">⭐ Star this repository if you found it helpful!

Built with ❤️ and Python

</div>