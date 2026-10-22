"""
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: train_models.py
Description: Trains, evaluates, and serializes K-Nearest Neighbors (KNN)
             and Naïve Bayes classifiers for Disease Prediction (Diabetes).
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# File paths
DATASET_PATH = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\datasets\diabetes.csv"
MODEL_DIR = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\models"

def main():
    print("=" * 60)
    print("      STEP 1: LOADING & INSPECTING DIABETES DATASET       ")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFeature Summary:")
    print(df.describe().T[['min', 'mean', '50%', 'max']])

    print("\n" + "=" * 60)
    print("      STEP 2: DATA PREPROCESSING & CLEANING               ")
    print("=" * 60)

    # In clinical data, zero values in Glucose, BloodPressure, SkinThickness,
    # Insulin, and BMI represent missing/invalid physiological data.
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print(f"Handling zero-value anomalies in columns: {zero_invalid_cols}")

    df_cleaned = df.copy()
    for col in zero_invalid_cols:
        # Replace 0 with NaN, then impute with column median
        num_zeros = (df_cleaned[col] == 0).sum()
        median_val = df_cleaned[df_cleaned[col] != 0][col].median()
        df_cleaned[col] = df_cleaned[col].replace(0, median_val)
        print(f" - Column '{col}': Imputed {num_zeros} zero values with median ({median_val:.1f})")

    # Separate features (X) and target label (y)
    X = df_cleaned.drop(columns=['Outcome'])
    y = df_cleaned['Outcome']

    # Train-Test Split (80% Train, 20% Test) with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nData split completed: Training samples = {len(X_train)}, Testing samples = {len(X_test)}")

    # Feature Scaling (Crucial for distance-based KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling applied using StandardScaler (Zero mean, Unit variance).")

    print("\n" + "=" * 60)
    print("      STEP 3: MODEL TRAINING & HYPERPARAMETER SETUP       ")
    print("=" * 60)

    # 1. K-Nearest Neighbors (Instance-Based Learning)
    # Using k=7 with distance-weighted voting
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2)
    knn.fit(X_train_scaled, y_train)
    print("1. K-Nearest Neighbors (KNN) trained successfully (k=7, distance weights).")

    # 2. Gaussian Naïve Bayes (Bayesian Learning)
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    print("2. Gaussian Naïve Bayes trained successfully.")

    # 3. Logistic Regression (Baseline Comparison)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    print("3. Logistic Regression trained successfully (Baseline comparison model).")

    print("\n" + "=" * 60)
    print("      STEP 4: MODEL EVALUATION & METRICS COMPUTATION     ")
    print("=" * 60)

    models = {
        "KNN": knn,
        "Naïve Bayes": nb,
        "Logistic Regression": lr
    }

    metrics_summary = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_summary[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
            "raw_acc": acc
        }

        print(f"\n--- Model: {name} ---")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  F1-Score : {f1 * 100:.2f}%")
        print(f"  Confusion Matrix (TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}):")
        print(f"    [[ {cm[0][0]:3d}, {cm[0][1]:3d} ]")
        print(f"     [ {cm[1][0]:3d}, {cm[1][1]:3d} ]]")

    # Identify best performing model
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]['raw_acc'])
    print("\n" + "=" * 60)
    print(f"   BEST MODEL IDENTIFIED: {best_model_name} ({metrics_summary[best_model_name]['accuracy']}%)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("      STEP 5: PERSISTING MODELS & METRICS ARTIFACTS      ")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(knn, os.path.join(MODEL_DIR, "knn_model.pkl"))
    joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metrics_file = os.path.join(MODEL_DIR, "disease_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics_summary,
            "best_model": best_model_name,
            "feature_names": list(X.columns)
        }, f, indent=4)

    print(f"Saved artifacts to directory: {MODEL_DIR}")
    print(" - knn_model.pkl")
    print(" - naive_bayes_model.pkl")
    print(" - logistic_regression_model.pkl")
    print(" - scaler.pkl")
    print(" - disease_metrics.json")
    print("\nModel training and evaluation successfully completed!")

if __name__ == "__main__":
    main()
