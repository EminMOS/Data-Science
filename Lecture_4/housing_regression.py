"""
Housing price prediction and classification script

Tasks:
1) Linear Regression for price prediction
2) Logistic Regression for classifying homes above the median price
3) Plots and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve, auc

# 1) Load dataset
df = pd.read_csv("houses.csv")

# 2) Target selection (tries common names, else last numeric column)
candidate_targets = ["price", "SalePrice", "median_house_value", "HousePrice", "target"]
target_col = None
for c in df.columns:
    if c.lower() in [t.lower() for t in candidate_targets]:
        target_col = c
        break

if target_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found. Cannot define a target variable.")
    target_col = numeric_cols[-1]

y = df[target_col]
X = df.drop(columns=[target_col])

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Preprocessing
numeric_features = selector(dtype_include=np.number)(X_train)
categorical_features = selector(dtype_exclude=np.number)(X_train)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ========== Linear Regression ==========
linreg_model = Pipeline(steps=[("preprocess", preprocess), ("model", LinearRegression())])
linreg_model.fit(X_train, y_train)
y_pred_reg = linreg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
print(f"Test MSE (Linear Regression): {mse:.4f}")

plt.figure()
plt.scatter(y_test, y_pred_reg)
min_val = min(np.min(y_test), np.min(y_pred_reg))
max_val = max(np.max(y_test), np.max(y_pred_reg))
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: True vs Predicted")
plt.show()

# Feature vs target plots for top-3 numeric correlations
if len(numeric_features) > 0:
    corr = df[numeric_features + [target_col]].corr()[target_col].drop(labels=[target_col])
    corr_abs = corr.abs().sort_values(ascending=False)
    top3_feats = corr_abs.head(min(3, len(corr_abs))).index.tolist()
    for feat in top3_feats:
        plt.figure()
        plt.scatter(df[feat], df[target_col])
        plt.xlabel(feat)
        plt.ylabel(target_col)
        plt.title(f"{feat} vs {target_col}")
        plt.show()

# ========== Logistic Regression ==========
median_price = y.median()
y_binary = (y > median_price).astype(int)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_binary, test_size=0.2, random_state=42)

logreg_model = Pipeline(steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])
logreg_model.fit(X_train_b, y_train_b)
y_pred_cls = logreg_model.predict(X_test_b)
acc = accuracy_score(y_test_b, y_pred_cls)
print(f"Test Accuracy (Logistic Regression): {acc:.4f}")

cm = confusion_matrix(y_test_b, y_pred_cls)
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Logistic Regression: Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.show()

y_scores = logreg_model.predict_proba(X_test_b)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_b, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression: ROC Curve")
plt.legend()
plt.show()

print(f"Binary threshold based on median price: {median_price:.3f}")
