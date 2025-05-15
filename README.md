# 🔌 Time Series Analysis and Forecasting for Household Power Consumption
📋 Project Overview
This comprehensive project analyzes household power consumption data through multiple approaches: SARIMA modeling, univariate and multivariate deep learning forecasting, and consumption classification. The project implements state-of-the-art time series techniques to predict future energy consumption patterns and classify usage levels.
🎯 Objectives

Perform exploratory data analysis on household power consumption data
Implement traditional time series forecasting with SARIMA
Develop deep learning models (MLP, CNN, LSTM, ConvLSTM) for univariate prediction
Extend to multivariate forecasting incorporating multiple electrical measurements
Create a classification system for consumption levels (Low, Medium, High)
Optimize models using Bayesian optimization (Optuna)
Track all experiments using MLflow with DagsHub integration

📊 Dataset
The dataset contains household power consumption measurements collected at minute-level granularity:

Time Period: December 2006 - November 2010
Original Granularity: 1-minute intervals
Processed Granularity: Daily aggregates
Final Dataset Size: 1,425 days (after cleaning)

Features:

Global_active_power: Total active power consumed
Global_reactive_power: Total reactive power consumed
Voltage: Average voltage
Global_intensity: Average current intensity
Sub_metering_1: Kitchen energy consumption
Sub_metering_2: Laundry room energy consumption
Sub_metering_3: Water heater and air conditioner

🛠️ Project Structure
├── data/
│   ├── raw/                    # Original minute-level data
│   └── processed/              # Daily aggregated data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_sarima.ipynb        # SARIMA modeling
│   ├── 03_univariate.ipynb    # Univariate deep learning
│   ├── 04_multivariate.ipynb  # Multivariate forecasting
│   └── 05_classification.ipynb # Consumption classification
├── src/
│   ├── data_preprocessing/
│   ├── models/
│   ├── visualization/
│   └── utils/
├── models/                     # Saved trained models
├── results/                    # Metrics and visualizations
└── mlruns/                    # MLflow tracking data


🔬 Methodology
1. Data Preprocessing

Handled missing values (1.25% of data)
Removed days with >70% missing data
Aggregated minute-level data to daily frequency
Applied MinMax scaling for neural networks

2. SARIMA Analysis

Decomposed time series into trend, seasonal, and residual components
Performed stationarity tests (ADF, KPSS)
Used auto_arima for optimal parameter selection
Model: SARIMA(3,0,2)(1,1,1)[7] with weekly seasonality

3. Univariate Forecasting
Implemented and compared four architectures:

MLP: Multi-layer perceptron with dropout regularization
CNN: 1D convolutional network for temporal pattern extraction
LSTM: Long Short-Term Memory for sequence modeling
ConvLSTM: Hybrid architecture combining CNN and LSTM

Best Results (Test RMSE):

LSTM Simple: 0.2271
ConvLSTM Stacked: 0.2295
MLP Light: 0.2315
CNN Simple: 0.2347

4. Multivariate Forecasting
Extended univariate models to incorporate all electrical measurements:

Used 7 features for prediction
Applied feature engineering (moving averages, standard deviations)
Implemented attention mechanisms for feature importance

5. Classification Analysis
Developed a three-class consumption classification system:

Classes: Low (0-40%), Medium (40-75%), High (75-100%)
Features: 18 engineered features including temporal patterns
Models: MLP, CNN, LSTM, CNN-LSTM, Random Forest, XGBoost, SVM

Classification Results:

Random Forest: 100% accuracy
XGBoost: 100% accuracy
MLP: 93.02% accuracy
CNN-LSTM: 88.71% accuracy

🔧 Technologies Used

Python 3.8+
Deep Learning: TensorFlow/Keras
Machine Learning: scikit-learn, XGBoost
Time Series: statsmodels, pmdarima
Optimization: Optuna
Experiment Tracking: MLflow + DagsHub
Visualization: matplotlib, seaborn, plotly
Data Processing: pandas, numpy

📈 Key Findings

Missing Data Patterns: Identified systematic gaps (5 days in August 2010)
Seasonality: Strong weekly patterns in consumption
Model Performance: LSTM architectures performed best for univariate forecasting
Feature Importance: Temporal features crucial for classification
Optimization: Optuna optimization improved LSTM performance by ~4%

📊 Results Visualization
All results are automatically logged to MLflow with:

Model parameters
Performance metrics
Training curves
Prediction visualizations
Feature importance plots

🔄 Future Improvements

Implement ensemble methods for forecasting
Add external features (weather, holidays)
Explore transformer architectures
Develop real-time prediction pipeline
Create interactive dashboard for visualization

👥 Contributors

Paco Tinoco
Juan Pedro Ley
Carlos Moreno