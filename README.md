# Predicting and Explaining Traffic Collision Severity in Toronto Using Explainable Machine Learning

**Author:** Nishi Bhavesh Patel | Student ID: 501356244
**Program:** Master of Data Science & Analytics (Full-Time)
**Supervisor:** Professor Mucahit Cevik
**Institution:** Toronto Metropolitan University

---

## Project Overview

This project develops and evaluates a machine learning framework to predict traffic collision severity in Toronto — classified as **Minor**, **Major**, or **Fatal** — using historical collision records merged with hourly weather data. Beyond prediction, the project applies **SHAP (SHapley Additive Explanations)** to interpret model decisions and **DBSCAN spatial clustering** to identify high-risk collision hotspots across the city, supporting evidence-based road safety planning.

During development, two methodological issues were identified and corrected: **outcome-correlated involvement-flag features that constituted data leakage**, and a **categorical-encoding error** in which SMOTE-NC resampled one-hot-encoded temporal features as independent binary columns instead of as single categorical variables. All results below reflect the dataset and models **after both corrections**.

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| Primary | [City of Toronto Open Data Portal](https://open.toronto.ca/catalogue/) | 809,034 traffic collision records (2014–2026) |
| Secondary | [Government of Canada Historical Climate Data](https://climate.weather.gc.ca/) | Hourly weather observations, Toronto City Station (ID: 31688) |

The two datasets were merged on a date-hour key at a **100% match rate**. After removing 131,978 records (16.3%) with invalid GPS coordinates (latitude/longitude = 0.0), the final working dataset contains **677,056 records**. After feature engineering and removal of the six leakage-affected columns identified during the data-leakage audit (four raw involvement flags plus two features derived from them), the final model input matrix has **33 features (21 categorical, 12 continuous)**.

---

## Project Structure

| Notebook | Description |
|---|---|
| `01_Data_Collection_Preparation.ipynb` | Loads raw collision data, fixes Unix timestamps, downloads and cleans weather data, merges both datasets |
| `02_Data_Cleaning_Feature_Engineering.ipynb` | Removes invalid records, creates the severity target variable, engineers new features (season, hour category, weather flags, etc.), audits and removes leakage-affected involvement-flag columns |
| `03_Exploratory_Data_Analysis.ipynb` | Produces 8 key visualizations covering temporal, spatial, and weather-related patterns |
| `04_Model_Training_Evaluation.ipynb` | Trains and compares 5 classification models, applies SMOTE-NC and 4 alternative balancing strategies, performs hyperparameter tuning, threshold analysis, and 5-fold cross-validation on the full corrected dataset |
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
3. **Data Leakage Audit** — Quantify and remove involvement-flag features found to be populated in a manner correlated with the severity outcome (Fatal rate 21.0× higher when `PEDESTRIAN` was set, 19.7× for `MOTORCYCLE`, 3.1× for `PASSENGER`, 2.9× for `BICYCLE`, relative to the dataset-wide Fatal rate)
4. **Exploratory Data Analysis** — Visualize temporal, spatial, and weather patterns
5. **Model Training & Evaluation** — Train 5 classifiers on the corrected feature set, address class imbalance via SMOTE-NC (with a categorical-encoding fix applied before resampling — see below) and 4 alternative strategies, tune hyperparameters, validate with 5-fold cross-validation on the full 677,056-record dataset
6. **Explainability & Geospatial Analysis** — SHAP interpretation of the corrected model and DBSCAN hotspot detection

### The SMOTE-NC Encoding Fix

`day_of_week`, `season`, and `hour_category` had already been one-hot encoded into separate binary columns before balancing. A direct check found that resampling these columns independently (rather than as a single categorical choice) corrupted **34.8% of day-of-week rows, 15.0% of hour-category rows, and 5.8% of season rows** in the synthetic training data. The fix collapses each one-hot group back into a single categorical column before SMOTE-NC runs, then re-expands it afterward — verified by a post-resampling check confirming zero invalid rows across all three groups.

---

## Models Compared (5-Fold Cross-Validation, Full 677,056-Record Dataset, Corrected Pipeline)

| Model | Accuracy | F1 Weighted | F1 Macro | ROC-AUC | PR-AUC (Fatal) |
|---|---|---|---|---|---|
| **XGBoost** | 0.8142 ± 0.0043 | 0.7851 ± 0.0016 | **0.3381 ± 0.0014** | 0.5465 ± 0.0027 | 0.0012 ± 0.0002 |
| Gradient Boosting | 0.6688 ± 0.0098 | 0.7106 ± 0.0063 | 0.3338 ± 0.0020 | 0.5355 ± 0.0021 | 0.0014 ± 0.0003 |
| Random Forest | 0.5501 ± 0.0061 | 0.6314 ± 0.0041 | 0.3057 ± 0.0018 | 0.5229 ± 0.0011 | **0.0017 ± 0.0004** |
| Neural Network | 0.4965 ± 0.0352 | 0.5811 ± 0.0303 | 0.2871 ± 0.0110 | 0.5135 ± 0.0051 | 0.0011 ± 0.0001 |
| Logistic Regression | 0.4280 ± 0.0101 | 0.5329 ± 0.0092 | 0.2668 ± 0.0034 | 0.5152 ± 0.0026 | 0.0010 ± 0.0001 |

A no-skill (uninformative) classifier's PR-AUC is approximately equal to the Fatal-class prevalence, roughly **0.0009** here — all five models score only marginally above that floor.

**No single model is an unqualified winner.** XGBoost has the highest F1 Macro and Accuracy, making it the strongest overall three-class model under the primary aggregate criterion — it is retained as this study's primary model. Random Forest has the highest Fatal-class PR-AUC (0.0017), making it the strongest model on the Fatal-specific discrimination metric that matters most for this problem. On the validation-split confusion matrix, XGBoost identifies **zero of 95** real Fatal collisions despite leading F1 Macro and Accuracy — a direct illustration of why PR-AUC (Fatal), not accuracy or F1 Macro, is treated as the primary lens on genuine Fatal-class ability in this study.

---

## Key Findings

- **Class Imbalance:** Fatal collisions represent only 0.1% of records, motivating a comparative analysis of 5 balancing strategies (SMOTE-NC, ADASYN, SMOTETomek, Random Undersampling, class-weighted/no resampling).
- **Fatal-Class Discrimination Remains Practically Weak:** After both corrections, Fatal-class PR-AUC ranges from 0.0010 to 0.0017 across all five model families — close in absolute terms to the ~0.0009 no-skill reference level. This supports the interpretation that the current predictor set (time, weather, location, neighbourhood frequency) provides limited Fatal-class separability under the modelling setup used here, without ruling out improvements from richer predictors or alternative approaches.
- **Threshold Tuning:** Lowering the Fatal classification threshold from 0.50 to 0.05 raises XGBoost's Fatal recall from 0.0% to 58.95%, but overall accuracy drops from 81.45% to 42.97%, and Fatal precision at 0.05 falls to just 0.11% (roughly 900 false flags per true catch). This is reported as evidence of a trade-off, not as an operational deployment recommendation — this study makes none.
- **Top Risk Factors (SHAP, Corrected Model):** `is_weekend`, `OCC_YEAR`, `OCC_MONTH`, and location features (`neighbourhood_freq`, `HOOD_158`) are the strongest model dependencies. SHAP explains what the fitted model relies on, not validated real-world causal risk factors — a caveat that matters given the model's limited genuine Fatal-class discriminative ability.
- **Geospatial Hotspots:** DBSCAN identified **36 distinct Fatal-collision hotspot clusters** across Toronto, led by Wexford/Maryvale (18 fatal collisions), West Humber-Clairville (17), and South Parkdale (14). Because 499 of 636 (78.5%) Fatal observations were classified as spatial noise rather than part of a cluster, these results are interpreted as a descriptive, supplementary spatial-prioritization layer rather than a comprehensive risk map. This analysis is unaffected by the leakage and encoding issues above, since it depends only on real GPS coordinates and the confirmed severity label.
- **Hyperparameter Tuning Is, At Best, a Mixed Intervention:** `RandomizedSearchCV`'s internal cross-validated score (0.93 for XGBoost, 0.79 for Random Forest) — computed on already-resampled data — does not reproduce on real, held-out validation data. Tuning decreases XGBoost's F1 Macro (0.337 → 0.318) with no recall change, and increases Random Forest's F1 Macro (0.308 → 0.339) while its Fatal recall collapses from 17.9% to 1.1%. Any future use of this pipeline should evaluate tuned models against held-out, unresampled data before adopting tuned parameters.

---

## Technologies Used

- **Language:** Python 3.x
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Class Imbalance Handling:** imbalanced-learn (SMOTE-NC, ADASYN, SMOTETomek, RandomUnderSampler)
- **Explainable AI:** SHAP
- **Geospatial Analysis:** Folium, DBSCAN (scikit-learn)
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

- **Fatal-class discrimination remains practically weak under the corrected feature set and modelling setup.** This is the central limitation of the current feature set, not a limitation of any one model choice, and it holds under both the leakage correction and the subsequent one-hot/SMOTE-NC encoding fix.
- **The predictive model, as currently specified, is not ready for operational deployment for Fatal-collision flagging.** With PR-AUC (Fatal) of 0.0010–0.0017 across all five tested model families, recommending threshold-based deployment is not supported by the corrected results, and this study makes none.
- Wind speed and visibility data were entirely unavailable from Toronto City Station ID 31688 (100% null values), limiting the weather feature set to temperature, humidity, dew point, and precipitation.
- The weather station represents city-wide conditions and may not reflect hyperlocal conditions at every collision site.
- The Fatal class is extremely small (0.1%, 636 records), limiting the model's ability to learn robust patterns for this category despite resampling techniques.
- The dataset only includes officially reported collisions; unreported minor incidents are not captured.
- This audit cannot rule out subtler, undetected leakage that produces a less conspicuous SHAP signal than the four removed involvement flags did.

---

## References

See the full reference list in the project report, including foundational works by Lundberg et al. (2020) on SHAP, Parsa et al. (2020) on XGBoost for crash severity, and Xie & Yan (2008) on spatial clustering of traffic accidents.
