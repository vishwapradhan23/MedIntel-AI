"""
=============================================================================
MedIntel AI: Symptom Triage Model Training & Evaluation
Script: train_triage.py
Description: Trains and serializes the primary Triage Model (TF-IDF with unigrams
             and bigrams + Logistic Regression with balanced class weights)
             and benchmarks against multiple baselines (MultinomialNB, Unigram-only,
             and optional local Sentence-Transformers). Evaluates rigorously on
             the hand-crafted test dataset and saves serialized models & metrics.
=============================================================================
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure data generator runs if dataset missing
import generate_triage_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "triage_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "triage_test.csv")

# Negation words that MUST NEVER be removed
NEGATION_WORDS = {
    "no", "not", "without", "never", "dont", "cant", "wont",
    "isnt", "arent", "neither", "nor", "none", "hardly", "barely",
    "scarcely", "nothing", "nowhere", "didnt", "hasnt", "havent"
}

def preprocess_text(text):
    """
    Triage Text Preprocessing:
    1. Handle non-string / missing values
    2. Convert to lowercase
    3. Normalize contractions (don't -> dont, can't -> cant)
    4. Remove punctuation, symbols, and digits
    5. Tokenize and normalize whitespace
    6. Retain all negation words strictly
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize common contractions
    text = re.sub(r"won['’]t", "wont", text)
    text = re.sub(r"can['’]t", "cant", text)
    text = re.sub(r"n['’]t", "nt", text)
    # Remove punctuation, symbols, digits
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def train_and_evaluate():
    """Execute training pipeline, compare baselines, and save models."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Ensure datasets exist
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print("Dataset files missing. Running generate_triage_data.py...")
        generate_triage_data.main()

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    print("=" * 65)
    print("      TRAINING SYMPTOM TRIAGE SPECIALTY CLASSIFIER        ")
    print("=" * 65)
    print(f"Loaded Training Set: {len(df_train)} rows across {df_train['specialty'].nunique()} specialties")
    print(f"Loaded Test Set    : {len(df_test)} rows across {df_test['specialty'].nunique()} specialties")

    # Apply preprocessing
    X_train_raw = df_train["text"].apply(preprocess_text)
    y_train = df_train["specialty"]

    X_test_raw = df_test["text"].apply(preprocess_text)
    y_test = df_test["specialty"]

    specialties_list = sorted(list(y_train.unique()))

    # =========================================================================
    # 2. PRIMARY MODEL: TF-IDF (Unigram + Bigram) + LogisticRegression (Balanced)
    # =========================================================================
    print("\n[1/4] Training Primary Model: TF-IDF (1,2) + LogisticRegression (balanced)...")
    primary_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_train_tfidf = primary_vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = primary_vectorizer.transform(X_test_raw)

    primary_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=2.0
    )
    primary_model.fit(X_train_tfidf, y_train)

    y_pred_primary = primary_model.predict(X_test_tfidf)
    acc_primary = float(accuracy_score(y_test, y_pred_primary))
    prec_primary = float(precision_score(y_test, y_pred_primary, average="weighted", zero_division=0))
    rec_primary = float(recall_score(y_test, y_pred_primary, average="weighted", zero_division=0))
    f1_macro_primary = float(f1_score(y_test, y_pred_primary, average="macro", zero_division=0))
    f1_weighted_primary = float(f1_score(y_test, y_pred_primary, average="weighted", zero_division=0))

    print(f"  -> Test Accuracy : {acc_primary * 100:.2f}%")
    print(f"  -> Macro F1      : {f1_macro_primary * 100:.2f}%")
    print(f"  -> Weighted F1   : {f1_weighted_primary * 100:.2f}%")

    # Per-class report for primary model
    class_rep = classification_report(y_test, y_pred_primary, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred_primary, labels=specialties_list).tolist()

    # =========================================================================
    # 3. BASELINE 1: Multinomial Naive Bayes on TF-IDF (1,2)
    # =========================================================================
    print("\n[2/4] Training Baseline 1: Multinomial Naive Bayes on TF-IDF (1,2)...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    acc_nb = float(accuracy_score(y_test, y_pred_nb))
    f1_macro_nb = float(f1_score(y_test, y_pred_nb, average="macro", zero_division=0))
    f1_weighted_nb = float(f1_score(y_test, y_pred_nb, average="weighted", zero_division=0))
    print(f"  -> Test Accuracy : {acc_nb * 100:.2f}% | Macro F1: {f1_macro_nb * 100:.2f}%")

    # =========================================================================
    # 4. BASELINE 2: Unigram-only TF-IDF + LogisticRegression
    # =========================================================================
    print("\n[3/4] Training Baseline 2: Unigram-only TF-IDF (1,1) + LogisticRegression...")
    unigram_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=1
    )
    X_train_unigram = unigram_vectorizer.fit_transform(X_train_raw)
    X_test_unigram = unigram_vectorizer.transform(X_test_raw)

    lr_unigram = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr_unigram.fit(X_train_unigram, y_train)
    y_pred_unigram = lr_unigram.predict(X_test_unigram)
    acc_unigram = float(accuracy_score(y_test, y_pred_unigram))
    f1_macro_unigram = float(f1_score(y_test, y_pred_unigram, average="macro", zero_division=0))
    f1_weighted_unigram = float(f1_score(y_test, y_pred_unigram, average="weighted", zero_division=0))
    print(f"  -> Test Accuracy : {acc_unigram * 100:.2f}% | Macro F1: {f1_macro_unigram * 100:.2f}%")

    # =========================================================================
    # 5. BASELINE 3 (OPTIONAL): Sentence-Transformers + LogisticRegression
    # =========================================================================
    sentence_trans_metrics = {"available": False, "note": "sentence-transformers not installed or offline"}
    print("\n[4/4] Checking Optional Baseline 3: Sentence-Transformers (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  -> Generating offline embeddings for train and test...")
        train_embeddings = st_model.encode(list(X_train_raw), show_progress_bar=False)
        test_embeddings = st_model.encode(list(X_test_raw), show_progress_bar=False)

        lr_embeddings = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_embeddings.fit(train_embeddings, y_train)
        y_pred_st = lr_embeddings.predict(test_embeddings)

        acc_st = float(accuracy_score(y_test, y_pred_st))
        f1_macro_st = float(f1_score(y_test, y_pred_st, average="macro", zero_division=0))
        f1_weighted_st = float(f1_score(y_test, y_pred_st, average="weighted", zero_division=0))
        sentence_trans_metrics = {
            "available": True,
            "accuracy": round(acc_st * 100, 2),
            "macro_f1": round(f1_macro_st * 100, 2),
            "weighted_f1": round(f1_weighted_st * 100, 2),
            "model_name": "all-MiniLM-L6-v2 + LogisticRegression"
        }
        print(f"  -> Test Accuracy : {acc_st * 100:.2f}% | Macro F1: {f1_macro_st * 100:.2f}%")
    except Exception as e:
        print(f"  -> Sentence-Transformers offline evaluation skipped: {e}")
        sentence_trans_metrics = {
            "available": False,
            "note": f"Sentence-Transformers not active locally ({str(e)[:80]})"
        }

    # =========================================================================
    # 6. Extract Top Salient Features per Specialty for Clinical Explainability
    # =========================================================================
    feature_names = primary_vectorizer.get_feature_names_out()
    top_features_per_specialty = {}
    for i, specialty in enumerate(primary_model.classes_):
        top_coef_indices = np.argsort(primary_model.coef_[i])[-10:][::-1]
        top_terms = [feature_names[idx] for idx in top_coef_indices]
        top_features_per_specialty[specialty] = top_terms

    # =========================================================================
    # 7. Serialize Artifacts and Metrics
    # =========================================================================
    model_path = os.path.join(MODEL_DIR, "triage_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "triage_vectorizer.pkl")
    metrics_path = os.path.join(MODEL_DIR, "triage_metrics.json")

    joblib.dump(primary_model, model_path)
    joblib.dump(primary_vectorizer, vectorizer_path)

    metrics_payload = {
        "primary_model": {
            "name": "Logistic Regression (Balanced) + TF-IDF (1,2)",
            "accuracy": round(acc_primary * 100, 2),
            "precision": round(prec_primary * 100, 2),
            "recall": round(rec_primary * 100, 2),
            "macro_f1": round(f1_macro_primary * 100, 2),
            "weighted_f1": round(f1_weighted_primary * 100, 2),
            "vocabulary_size": len(primary_vectorizer.vocabulary_),
            "per_class_metrics": {
                spec: {
                    "precision": round(class_rep[spec]["precision"] * 100, 2),
                    "recall": round(class_rep[spec]["recall"] * 100, 2),
                    "f1_score": round(class_rep[spec]["f1-score"] * 100, 2),
                    "support": class_rep[spec]["support"]
                }
                for spec in specialties_list if spec in class_rep
            },
            "confusion_matrix": {
                "labels": specialties_list,
                "matrix": conf_mat
            },
            "top_features": top_features_per_specialty
        },
        "baselines_comparison": {
            "multinomial_nb": {
                "name": "Multinomial Naive Bayes (TF-IDF 1,2)",
                "accuracy": round(acc_nb * 100, 2),
                "macro_f1": round(f1_macro_nb * 100, 2),
                "weighted_f1": round(f1_weighted_nb * 100, 2)
            },
            "unigram_only": {
                "name": "Logistic Regression (TF-IDF 1,1 Unigram Only)",
                "accuracy": round(acc_unigram * 100, 2),
                "macro_f1": round(f1_macro_unigram * 100, 2),
                "weighted_f1": round(f1_weighted_unigram * 100, 2)
            },
            "sentence_transformers": sentence_trans_metrics
        },
        "academic_note": "Evaluated on strictly held-out hand-written test set (data/triage_test.csv). Metrics are indicative decision support evaluations."
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n" + "=" * 65)
    print("      SAVED SERIALIZED TRIAGE ARTIFACTS                  ")
    print("=" * 65)
    print(f" -> Model: {model_path}")
    print(f" -> Vectorizer: {vectorizer_path}")
    print(f" -> Metrics: {metrics_path}")
    print("Triage model training and benchmarking completed successfully!")

    return metrics_payload

if __name__ == "__main__":
    train_and_evaluate()
