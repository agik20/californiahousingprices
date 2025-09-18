# California Housing Prices – Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

A comprehensive machine learning project for predicting California housing prices using census data. This project demonstrates a complete end-to-end ML pipeline from exploratory data analysis to model deployment.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Web Application](#-web-application)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)

## 🎯 Project Overview

This repository contains an exploratory data analysis (EDA) and machine learning implementation on the **California Housing Prices dataset** from the 1990 California census. The project follows best practices in data science workflow and demonstrates:

- Comprehensive data exploration and visualization
- Data preprocessing and feature engineering
- Multiple machine learning model implementation
- Model evaluation and comparison
- Web application deployment for predictions

**Reference:** This dataset is famously used in *"Hands-On Machine Learning with Scikit-Learn and TensorFlow"* by Aurélien Géron.

## 📊 Dataset

**Source:** Originally from R. Kelley Pace and Ronald Barry (1997), modified by Aurélien Géron.

**Features:**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of housing in block
- `total_rooms` - Total number of rooms in block
- `total_bedrooms` - Total number of bedrooms in block
- `population` - Population in block
- `households` - Number of households in block
- `median_income` - Median income in block (scaled)
- `ocean_proximity` - Categorical proximity to ocean
- `median_house_value` - Target variable (median house value in USD)

## 💻 Installation

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
docker run -d -p 80:5000 house-price-app
```

## 📁 Project Structure

```
californiahousinprice/
├── data/                    # Dataset files
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Source code
│   ├── data_processing.py   # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── models.py           # ML model implementations
│   └── visualization.py    # Plotting functions
├── app.py                  # Flask web application
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
└── Dockerfile             # Container configuration
```

## 🔬 Methodology

### 1. Data Loading & Exploration
- Initial dataset inspection and statistical summary
- Missing value analysis (handled in `total_bedrooms`)
- Distribution analysis of numerical features

### 2. Data Preprocessing
- Handling missing values using median imputation
- Categorical encoding (`ocean_proximity` → one-hot encoding)
- Standard scaling of numerical variables

### 3. Feature Engineering
Created informative derived features:
- `rooms_per_household` - Average rooms per household
- `bedrooms_per_room` - Bedroom to room ratio
- `population_per_household` - Population density metric

### 4. Modeling Approach
- **Baseline Model**: Linear Regression
- **Advanced Model**: Random Forest Regressor with GridSearchCV
- **Evaluation Metrics**: MAE, MSE, RMSE, R²

### 5. Model Evaluation
- Cross-validation performance assessment
- Error analysis and residual examination
- Feature importance interpretation

## 📈 Results

### Model Performance
| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | 48,660.76 | 4,530,030,653.76 | 67,305.50 | 0.6687 |
| Random Forest | 32,187.41 | 2,424,929,897.71 | 49,243.58 | 0.8227 |
| XGBoost | 31827.2788 | 2316064795.3990 | 48125.5109 | 0.8306 |

### Visualizations

<div align="center">
  
**Before Feature Engineering**
<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/4a5631de-ff7e-421d-9ce7-5565f6b8d3d8" width="400" alt="Correlation Matrix Before">
<br><em>Correlation Matrix (Original Features)</em>
</td>
<td align="center">
<img src="https://github.com/user-attachments/assets/c564791e-315a-4c18-be87-68a54806d6ba" width="400" alt="Feature Distribution Before">
<br><em>Feature Distribution (Original Features)</em>
</td>
</tr>
</table>

**After Feature Engineering**
<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/4d5e4b90-2962-4a24-b33a-99878c03bfa9" width="400" alt="Correlation Matrix After">
<br><em>Correlation Matrix (Engineered Features)</em>
</td>
<td align="center">
<img src="https://github.com/user-attachments/assets/b8fc824d-b045-475f-a036-3494ef3c82fe" width="400" alt="Feature Distribution After">
<br><em>Feature Distribution (Engineered Features)</em>
</td>
</tr>
</table>

**Final Analysis**
<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/25ccadcf-e511-44f6-9d56-23e92d17805f" width="400" alt="Final Correlation">
<br><em>Final Correlation Analysis</em>
</td>
<td align="center">
<img src="https://github.com/user-attachments/assets/96dfaf1f-cfa6-4d21-a9e3-cd65e07bca75" width="400" alt="Scatter Plots">
<br><em>Predictor-Target Relationships</em>
</td>
</tr>
</table>

</div>

## 🌐 Web Application

The project includes a Flask web application for interactive predictions:

### Local Deployment
```bash
python app.py
```
Access at: http://localhost:5000

### Docker Deployment
```bash
docker build -t house-price-app .
docker run -d -p 80:5000 house-price-app
```
Access at: http://localhost

## 🔍 Key Insights

1. **Income Dominance**: Median income shows the strongest correlation with house prices (≈0.69)
2. **Geographic Influence**: Coastal properties command premium prices, with "Near Ocean" and "Near Bay" locations being most valuable
3. **Feature Engineering Value**: Derived ratio features (rooms per household, etc.) provided more predictive power than raw counts
4. **Model Performance**: Random Forest significantly outperformed Linear Regression (R²: 0.82 vs 0.67), capturing non-linear relationships
5. **Spatial Patterns**: Clear geographic clustering of housing values, with higher prices concentrated in coastal regions

## 🚀 Future Work

- **Advanced Models**: Experiment with gradient boosting algorithms (XGBoost, LightGBM, CatBoost)
- **Hyperparameter Optimization**: Implement Bayesian optimization or more extensive grid search
- **Geospatial Analysis**: Integrate interactive mapping with Folium or GeoPandas
- **Feature Expansion**: Incorporate additional external data sources (crime rates, school quality, etc.)
- **Deployment Enhancement**: Containerize with Kubernetes for scalable deployment
- **Monitoring**: Implement model performance monitoring and drift detection

## 📚 References

1. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn and TensorFlow.
2. Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. Statistics & Probability Letters.
3. Glaeser, E. L., & Gyourko, J. (2008). Housing Dynamics. Harvard Institute of Economic Research.

---

**Note**: This project serves as an educational example of a complete machine learning pipeline from data exploration to deployment. The code and methodologies can be adapted for similar regression problems in real estate and other domains.

For questions or contributions, please open an issue or submit a pull request.
