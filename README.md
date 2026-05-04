# 💰 Salary Prediction ML Pipeline

 End-to-end machine learning project for predicting developer salaries from survey data, focused on building robust, interpretable pipelines for tabular data.

The project focuses on building a **clean, reproducible ML pipeline** with proper validation, feature engineering, and model selection.

---

## 🚀 Project overview

We model annual salary (`annual.pay.usd`) as a regression problem using structured survey data containing demographics, experience, and multi-select technology fields.

Key challenges:
- Highly skewed target variable
- Multi-value categorical features (e.g. technologies)
- Large number of sparse features
- Risk of data leakage during preprocessing

---

## ⚙️ Pipeline

### 1. Feature Engineering
- Custom handling of **multi-select columns** (multi-hot encoding + frequency filtering)
- Extraction of counts and aggregated signals
- Removal of rare categories (`min_freq`, `top_k`)

### 2. Target Processing
- Log transformation: `y = log(salary)`
- Predictions converted back using `exp()`
- Filtering unrealistic salary values to stabilize RMSE

### 3. Preprocessing
- Configurable:
  - Imputation (`mean`, `median`, etc.)
  - Scaling (`standard`, `robust`, `minmax`)
- Implemented using `ColumnTransformer` + `Pipeline`

### 4. Validation strategy
- **5-fold cross-validation**
- Fully leakage-safe:
  - Feature engineering fitted inside each fold
  - No information from validation folds leaks into training

### 5. Models tested
- Linear Regression
- Ridge / Lasso / Elastic Net
- KNN
- SVR (RBF kernel)
- Random Forest
- Gradient Boosting

### 6. Hyperparameter tuning
- Grid-style tuning over:
  - Model parameters
  - Preprocessing choices
  - Multi-select encoding parameters

### 7. Ensemble
- Stacking was tested
- Final decision: **best single model outperformed ensemble**

---

## 🏆 Final model

Selected based on:
- Lowest **cross-validated RMSE**
- Stable performance across folds

The final model is retrained on the full dataset before generating predictions.

---

## 📊 Evaluation

- Metric: **RMSE (Root Mean Squared Error)**
- Computed on original salary scale (USD)
- Validation results closely match leaderboard performance

---

## 🛠️ Tech stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib / seaborn (EDA)

## 📁 Structure

salary-regression/
│
├── test.csv
├── train.csv
├── salary_regression.ipynb   # main notebook (EDA + modeling)
├── submission.csv            # final predictions
└── README.md

## ⚠️ Notes

- CV scores are estimates — leaderboard results may vary slightly
- Filtering extreme salaries improves stability but may reduce performance on edge cases
- Multi-select encoding keeps only frequent categories

---

## 📌 Takeaways

- Proper validation (no leakage) matters more than complex models
- Feature engineering on tabular data is critical
- Simpler models can outperform ensembles when well-tuned


Link to the Kaggle competition: https://www.kaggle.com/competitions/ml-1-2026-task-2-developer-salary-prediction-regression
