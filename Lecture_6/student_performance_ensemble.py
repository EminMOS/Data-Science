import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              BaggingClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              StackingClassifier,
                              VotingClassifier)

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import joblib


CSV_PLACEHOLDER = "student-mat.csv"  # change if needed
TARGET_COL = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.20
HIGH_THRESHOLD = 10  # threshold for high performance (you can change to median or other)


# Utility functions
def load_data(path=CSV_PLACEHOLDER):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Please place the dataset file there or change the path.")
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path, sep=';')
    return df

def binarize_target(df, target_col=TARGET_COL, threshold=HIGH_THRESHOLD, strategy="threshold"):
    """
    Convert G3 to binary label. strategy:
      - "threshold": label = 1 if G3 >= threshold else 0
      - "median": label = 1 if G3 >= median else 0
    """
    if strategy == "median":
        thresh = df[target_col].median()
    else:
        thresh = threshold
    y = (df[target_col] >= thresh).astype(int)
    return y, thresh

def get_feature_names_from_column_transformer(ct, numeric_features, categorical_features):
    """
    After ColumnTransformer, extract output feature names (works for sklearn >= 0.23).
    """
    feature_names = []
    # numeric features pass-through or scaler
    feature_names.extend(numeric_features)
    # categorical get names from OneHotEncoder
    # find transformer for categorical_features
    for name, transformer, cols in ct.transformers_:
        if name == 'cat':
            # transformer might be a pipeline with OneHotEncoder at the end
            ohe = transformer.named_steps['ohe'] if hasattr(transformer, 'named_steps') else transformer
            try:
                cat_names = list(ohe.get_feature_names_out(cols))
            except Exception:
                # fallback
                cat_names = []
                for col in cols:
                    cat_names.append(col)
            feature_names.extend(cat_names)
    return feature_names

def evaluate_model(model, X_test, y_test, model_name="model", show_cm=True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    if show_cm:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low','High'], yticklabels=['Low','High'])
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix: {model_name}')
        plt.show()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def plot_feature_importances(model, feature_names, model_name="Model"):
    """
    Plot feature importances when available (tree-based classifiers).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_') and model.coef_.ndim == 1:
        importances = np.abs(model.coef_)
    else:
        print(f"Model {model_name} has no attribute feature_importances_ or coef_. Skipping.")
        return
    # align lengths
    if len(importances) != len(feature_names):
        print("Length mismatch between importances and feature names; skipping feature importance plot.")
        return
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:30]
    plt.figure(figsize=(8,6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title(f"Top feature importances — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main pipeline
# ---------------------------
def build_preprocessing_pipeline(df, target_col=TARGET_COL):
    """
    Detect numeric and categorical columns automatically and build a ColumnTransformer
    with SimpleImputer + StandardScaler for numeric, and SimpleImputer + OneHotEncoder for categorical.
    """
    X = df.drop(columns=[target_col])
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

    # Imputers and transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    return preprocessor, numeric_features, categorical_features

def main(csv_path=CSV_PLACEHOLDER):
    # Load
    df = load_data(csv_path)
    print("Raw shape:", df.shape)
    # Optional: quick EDA
    print(df.head())
    print(df[TARGET_COL].describe())

    # Binarize target
    y, thresh = binarize_target(df, TARGET_COL, threshold=HIGH_THRESHOLD, strategy="threshold")
    print(f"Using threshold for high performance: {thresh}")

    # Preprocessing
    preprocessor, numeric_feats, categorical_feats = build_preprocessing_pipeline(df, TARGET_COL)

    # Prepare X (drop target) and split
    X = df.drop(columns=[TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE, stratify=y)
    print("Train/test split:", X_train.shape, X_test.shape)

    # Build pipelines for models where preprocessing is required
    # Baseline: Logistic Regression
    pipe_logreg = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    # Baseline: Decision Tree
    pipe_dt = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    # Bagging: Bagged Decision Trees
    bagging = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                  n_estimators=50, random_state=RANDOM_STATE))
    ])

    # Random Forest (tree-based ensemble)
    rf_pipe = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])

    # Boosting: GradientBoosting and AdaBoost
    gb_pipe = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])

    ada_pipe = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ])

    # Stacking: combine logistic, random forest, gradient boosting
    estimators = [
        ('lr', pipe_logreg),
        ('rf', rf_pipe),
        ('gb', gb_pipe)
    ]
    # Note: StackingClassifier expects estimators without full pipelines that contain preprocessors duplicating work.
    # A practical approach: use a single preprocessor + stacking on already preprocessed features.
    # So we build a preprocessing transformer separately and create classifiers that accept raw numeric arrays.
    # To keep simple and avoid duplication, we will create a pipeline that preprocesses once and then a StackingClassifier.
    stacking_clf = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', StackingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
        ], final_estimator=LogisticRegression(), n_jobs=-1))
    ])

    # Voting classifier — an alternative stacking-like ensemble (hard voting)
    voting = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('dt', DecisionTreeClassifier(random_state=RANDOM_STATE))
        ], voting='hard'))
    ])

    models = {
        "LogisticRegression": pipe_logreg,
        "DecisionTree": pipe_dt,
        "Bagging_DT": bagging,
        "RandomForest": rf_pipe,
        "GradientBoosting": gb_pipe,
        "AdaBoost": ada_pipe,
        "Stacking": stacking_clf,
        "Voting": voting
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        res = evaluate_model(model, X_test, y_test, model_name=name)
        results[name] = res

    # Feature importance example for RandomForest and GradientBoosting
    # Need feature names after preprocessing
    feature_names = get_feature_names_from_column_transformer(preprocessor, numeric_feats, categorical_feats)
    # Access the trained tree model inside the pipeline
    try:
        rf_model = rf_pipe.named_steps['clf']
        # Need to get importances from rf_model but ensure features align: the preprocessor was fit as part of rf_pipe when we fit it earlier
        plot_feature_importances(rf_model, feature_names, model_name="RandomForest")
    except Exception as e:
        print("Could not plot RandomForest importances:", e)

    try:
        gb_model = gb_pipe.named_steps['clf']
        plot_feature_importances(gb_model, feature_names, model_name="GradientBoosting")
    except Exception as e:
        print("Could not plot GradientBoosting importances:", e)

    # Summarize results in a DataFrame
    res_df = pd.DataFrame(results).T
    print("\nSummary metrics across models:")
    print(res_df)

    # Optionally, persist the best model by F1
    best_model_name = res_df['f1'].idxmax()
    print(f"Best model by F1: {best_model_name}")
    best_model = models[best_model_name]
    joblib.dump(best_model, "best_student_model.joblib")
    print("Saved best model to best_student_model.joblib")

if __name__ == '__main__':
    # Use default CSV placeholder. Change the path here if your CSV has a different filename.
    try:
        main(csv_path=CSV_PLACEHOLDER)
    except Exception as e:
        print("Error while running main():", e)
        print("If the dataset file is named differently, change CSV_PLACEHOLDER at the top of the script or pass a path on the command line.")
