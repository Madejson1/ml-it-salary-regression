# Presentation Script — Developer Salary Prediction
# Duration: ~3 minutes 30 seconds (~525 words at 150 wpm)

---

## SLIDE 1 — Title  [~15 seconds]

"Good morning. Today I'll be presenting our solution to the ML1 salary prediction task.
The goal was to predict annual developer compensation from survey data, evaluated using
Root Mean Square Error. The key constraint was that we could only use ML1-syllabus models —
so no gradient boosting or neural networks."

---

## SLIDE 2 — Problem & Data  [~30 seconds]

"Our dataset has around 2 500 training observations and 628 test rows. What makes this
problem challenging is the nature of the features: many columns are multi-select survey
fields — for example, a developer might list ten programming languages or five cloud
platforms in a single cell. The target variable, annual salary, is also heavily skewed —
a handful of entries show salaries close to zero or above a million dollars.

Before touching any model, we made three foundational decisions: encode multi-select
columns as multi-hot binary features plus a count, model the target in log-space to
stabilise variance, and filter the training data to the range 1 000 to 300 000 USD
to remove the extreme outliers."

---

## SLIDE 3 — Feature Engineering  [~35 seconds]

"The feature engineering pipeline works as follows. We first extract the multi-select
columns into individual binary indicators — keeping only categories that appear at least
100 times, and only the top 10 per column, to avoid noise. We also compute a count
feature for each column as a breadth-of-experience proxy. On top of that, we add two
derived features: years of non-professional coding and the share of professional
experience relative to total coding years.

Most importantly, we apply smoothed K-fold target encoding to eight high-cardinality
categorical columns — including region, industry, company size, and developer role.
This replaces each category with a smoothed mean of log-salary, computed out-of-fold
to avoid leakage."

---

## SLIDE 4 — Model Comparison  [~45 seconds]

"We compared seven models under 5-fold cross-validation. On the raw, unfiltered data,
all models performed extremely poorly — RMSE around 85 to 93 thousand. The issue was
clear: a handful of extreme salary rows dominated the squared error.

After restricting training to the 1 000 to 300 000 range, RMSE dropped dramatically.
SVR with an RBF kernel became the best ML1-allowed model at around 36 600 RMSE.
Random Forest and Gradient Boosting scored around 34 400 and 34 700 respectively, but
those are outside the ML1 syllabus, so SVR was our primary candidate.

Linear models — Ridge, Lasso, and Elastic Net — stayed above 38 000, confirming that
the salary signal is non-linear and regularised linear regression is insufficient here."

---

## SLIDE 5 — Hyperparameter Tuning  [~35 seconds]

"We ran three tuning rounds for SVR. First, a preprocessing sweep over scalers, imputers,
and multi-select parameters — standard scaling and mean imputation came out on top.
Second, a grid search over C, epsilon, and gamma — the best configuration was C equals 1,
epsilon 0.03, gamma auto. Third, we ablated target encoding, progressively adding
categorical columns; the full eight-column set gave the best result.

Across these three rounds, CV RMSE dropped from 36 600 all the way down to 31 284 —
a 5 400-point improvement purely from tuning and feature engineering, with the same
base model."

---

## SLIDE 6 — Final Model & Results  [~30 seconds]

"The final pipeline filters training data to the salary range, applies multi-hot and
target encoding, scales features, fits the SVR on log-salary, and exponentiates
predictions back to USD before writing the submission file.

The 5-fold CV RMSE on the filtered evaluation is 31 284. The expected leaderboard
score is roughly 33 to 34 thousand. There is a structural 2 to 3 thousand gap because
we apply the salary filter before splitting into folds, which prevents the model from
ever validating on extreme test-set salaries — making CV slightly optimistic."

---

## SLIDE 7 — Summary  [~20 seconds]

"To summarise: SVR with RBF kernel is the best ML1-allowed model for this task.
The two biggest wins were salary range filtering, which alone cut RMSE by 55 thousand,
and smoothed target encoding, which contributed an additional 5 thousand reduction.
Approaches that did not work include winsorization and blending SVR with Ridge.
We expect a leaderboard score in the 33 to 34 thousand range. Thank you."

---
# END — total approx. 210 seconds (3 min 30 sec)
