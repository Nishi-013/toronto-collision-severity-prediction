# Predicting and Explaining Traffic Collision Severity in Toronto Using Explainable Machine Learning 

**Author:** Nishi Bhavesh Patel | Student ID: 501356244  
**Program:** Master of Data Science & Analytics (Full-Time)  
**Supervisor:** Professor Mucahit Cevik  
**Institution:** Toronto Metropolitan University

---

## Project Overview

This project develops and evaluates a machine learning framework to predict traffic collision severity in Toronto — classified as **Minor**, **Major**, or **Fatal** — using historical collision records merged with hourly weather data. Beyond prediction, the project applies **SHAP (SHapley Additive Explanations)** to interpret model decisions and **DBSCAN spatial clustering** to identify high-risk collision hotspots across the city, supporting evidence-based road safety planning.

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| Primary | [City of Toronto Open Data Portal](https://open.toronto.ca/catalogue/) | 809,034 traffic collision records (2014–2026) |
| Secondary | [Government of Canada Historical Climate Data](https://climate.weather.gc.ca/) | Hourly weather observations, Toronto City Station (ID: 31688) |

After cleaning and removing records with invalid GPS coordinates, the final working dataset contains **677,056 records** with **48 engineered features**.

---

## Project Structure

| Notebook | Description |
|---|---|
| `01_Data_Collection_Preparation.ipynb` | Loads raw collision data, fixes Unix timestamps, downloads and cleans weather data, merges both datasets |
| `02_Data_Cleaning_Feature_Engineering.ipynb` | Removes invalid records, creates the severity target variable, engineers new features (season, hour category, weather flags, etc.) |
| `03_Exploratory_Data_Analysis.ipynb` | Produces 8 key visualizations covering temporal, spatial, and weather-related patterns |
| `04_Model_Training_Evaluation.ipynb` | Trains and compares 5 classification models, applies SMOTE and 5 balancing strategies, performs hyperparameter tuning, threshold analysis, and cross-validation |
| `05_SHAP_Explainability_Geospatial.ipynb` | Computes SHAP values for model explainability, builds collision hotspot maps, runs DBSCAN spatial clustering |

---

## Target Variable

| Class | Label | Records | Percentage |
|---|---|---|---|
| 0 | Minor (property damage only) | 580,492 | 85.7% |
| 1 | Major (one or more injuries) | 95,928 | 14.2% |
| 2 | Fatal (one or more fatalities) | 636 | 0.1% |

---

## Methodology

1. **Data Collection & Preparation** — Merge collision and weather data on date/hour (100% match rate)
2. **Data Cleaning & Feature Engineering** — Handle missing values, derive new features, one-hot encode categoricals
3. **Exploratory Data Analysis** — Visualize temporal, spatial, and weather patterns
4. **Model Training & Evaluation** — Train 5 classifiers, address class imbalance, tune hyperparameters, validate with 5-fold CV
5. **Explainability & Geospatial Analysis** — SHAP interpretation and DBSCAN hotspot detection

---

## Models Compared

| Model | Accuracy | F1 Weighted | F1 Macro | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **0.9002** | **0.8920** | **0.5193** | **0.7839** |
| Random Forest | 0.8972 | 0.8911 | 0.5148 | 0.7809 |
| Gradient Boosting | 0.8979 | 0.8904 | 0.5126 | 0.7823 |
| Neural Network | 0.8700 | 0.8731 | 0.4973 | 0.7624 |
| Logistic Regression | 0.8652 | 0.8663 | 0.4765 | 0.7452 |

**Best Model: XGBoost**

---

## Key Findings

- **Class Imbalance:** Fatal collisions represent only 0.1% of records, motivating a comparative analysis of 5 balancing strategies (SMOTE, ADASYN, SMOTETomek, Random Undersampling, class-weighted loss).
- **Threshold Tuning:** Lowering the Fatal classification threshold from 0.50 to 0.05 improves Fatal recall from 5.2% to 36.5%, at the cost of overall accuracy dropping from 90.0% to 81.6% — an important precision-recall tradeoff for road safety applications.
- **Top Risk Factors (SHAP):** Year, season (Spring/Fall), neighbourhood, day of week (Friday), and pedestrian involvement are the strongest predictors of fatal outcomes.
- **Geospatial Hotspots:** DBSCAN identified **36 distinct fatal collision hotspot zones** across Toronto, with Wexford/Maryvale and West Humber-Clairville showing the highest fatal collision counts.

---

## Technologies Used

- **Language:** Python 3.x
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Class Imbalance Handling:** imbalanced-learn (SMOTE, ADASYN, SMOTETomek, RandomUnderSampler)
- **Explainable AI:** SHAP
- **Geospatial Analysis:** Folium, DBSCAN
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebooks, VS Code

---

## How to Run

1. Clone this repository
2. Place the following folder structure in your working directory:
   ```
   MRP - Final Sem/
   ├── Data/
   ├── Models/
   ├── Plots/
   └── Maps/
   ```
3. Install required packages:
   ```
   pip install pandas numpy scikit-learn xgboost imbalanced-learn shap folium matplotlib seaborn requests
   ```
4. Run the notebooks in order: `01` → `02` → `03` → `04` → `05`

---

## Limitations

- Wind speed and visibility data were unavailable from the Toronto City weather station
- The weather station represents city-wide conditions and may not reflect hyperlocal conditions at every collision site
- The Fatal class is extremely small (0.1%), limiting the model's ability to learn robust patterns for this category despite resampling techniques
- The dataset only includes officially reported collisions; unreported minor incidents are not captured

---

## References

See the full reference list in the project report, including foundational works by Lundberg et al. (2020) on SHAP, Parsa et al. (2020) on XGBoost for crash severity, and Xie & Yan (2008) on spatial clustering of traffic accidents.


