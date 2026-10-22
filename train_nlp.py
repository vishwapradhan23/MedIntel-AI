"""
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: train_nlp.py
Description: Trains, evaluates, and serializes NLP TF-IDF vectorizers
             and Multinomial Naïve Bayes classifiers for Patient Feedback
             Sentiment Analysis (3-Class) & Topic Classification (7-Domain).
=============================================================================
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# File Paths
SENTIMENT_DATASET_PATH = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\datasets\sentiment.csv"
TOPIC_DATASET_PATH = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\datasets\topic_feedback.csv"
MODEL_DIR = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\models"

def preprocess_text(text):
    """
    NLP Preprocessing Pipeline:
    1. Handle missing/non-string values
    2. Convert to lowercase
    3. Remove punctuation, special characters, and digits
    4. Tokenize by whitespace
    5. Strip trailing spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return " ".join(tokens)

def main():
    print("=" * 60)
    print("   MODULE 3: TRAINING PATIENT FEEDBACK SENTIMENT MODEL    ")
    print("=" * 60)

    if not os.path.exists(SENTIMENT_DATASET_PATH):
        raise FileNotFoundError(f"Sentiment dataset missing at {SENTIMENT_DATASET_PATH}")

    df_sent = pd.read_csv(SENTIMENT_DATASET_PATH)
    print(f"Loaded Sentiment Dataset: {len(df_sent)} records")
    print("Class Distribution:\n", df_sent['sentiment'].value_counts())

    # Preprocess text
    df_sent['cleaned_text'] = df_sent['feedback'].apply(preprocess_text)

    X_sent = df_sent['cleaned_text']
    y_sent = df_sent['sentiment']

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sent, y_sent, test_size=0.20, random_state=42, stratify=y_sent
    )

    # TF-IDF Vectorization for Sentiment
    sent_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_train_s_vec = sent_vectorizer.fit_transform(X_train_s)
    X_test_s_vec = sent_vectorizer.transform(X_test_s)
    print(f"TF-IDF Sentiment Vocabulary Size: {len(sent_vectorizer.vocabulary_)} features")

    # Multinomial Naïve Bayes Classifier for Sentiment
    sentiment_model = MultinomialNB(alpha=0.1)
    sentiment_model.fit(X_train_s_vec, y_train_s)

    # Evaluation
    y_pred_s = sentiment_model.predict(X_test_s_vec)
    sent_acc = float(accuracy_score(y_test_s, y_pred_s))
    sent_prec = float(precision_score(y_test_s, y_pred_s, average='weighted'))
    sent_rec = float(recall_score(y_test_s, y_pred_s, average='weighted'))
    sent_f1 = float(f1_score(y_test_s, y_pred_s, average='weighted'))

    print(f"Sentiment Model Metrics:")
    print(f" - Accuracy : {sent_acc * 100:.2f}%")
    print(f" - Precision: {sent_prec * 100:.2f}%")
    print(f" - Recall   : {sent_rec * 100:.2f}%")
    print(f" - F1-Score : {sent_f1 * 100:.2f}%")

    print("\n" + "=" * 60)
    print("   MODULE 4: TRAINING HEALTHCARE TOPIC CLASSIFIER MODEL  ")
    print("=" * 60)

    if not os.path.exists(TOPIC_DATASET_PATH):
        raise FileNotFoundError(f"Topic dataset missing at {TOPIC_DATASET_PATH}")

    df_topic = pd.read_csv(TOPIC_DATASET_PATH)
    print(f"Loaded Topic Dataset: {len(df_topic)} records")
    print("Topic Classes Distribution:\n", df_topic['topic'].value_counts())

    df_topic['cleaned_text'] = df_topic['feedback'].apply(preprocess_text)

    X_topic = df_topic['cleaned_text']
    y_topic = df_topic['topic']

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_topic, y_topic, test_size=0.20, random_state=42, stratify=y_topic
    )

    # TF-IDF Vectorization for Topic Classification
    topic_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_train_t_vec = topic_vectorizer.fit_transform(X_train_t)
    X_test_t_vec = topic_vectorizer.transform(X_test_t)
    print(f"TF-IDF Topic Vocabulary Size: {len(topic_vectorizer.vocabulary_)} features")

    # Multinomial Naïve Bayes for Multi-Class Topic Classification
    topic_model = MultinomialNB(alpha=0.1)
    topic_model.fit(X_train_t_vec, y_train_t)

    # Evaluation
    y_pred_t = topic_model.predict(X_test_t_vec)
    topic_acc = float(accuracy_score(y_test_t, y_pred_t))
    topic_prec = float(precision_score(y_test_t, y_pred_t, average='weighted'))
    topic_rec = float(recall_score(y_test_t, y_pred_t, average='weighted'))
    topic_f1 = float(f1_score(y_test_t, y_pred_t, average='weighted'))

    print(f"Topic Model Metrics:")
    print(f" - Accuracy : {topic_acc * 100:.2f}%")
    print(f" - Precision: {topic_prec * 100:.2f}%")
    print(f" - Recall   : {topic_rec * 100:.2f}%")
    print(f" - F1-Score : {topic_f1 * 100:.2f}%")

    print("\n" + "=" * 60)
    print("      SAVING SERIALIZED NLP MODELS & METRICS              ")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(sentiment_model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(sent_vectorizer, os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl"))
    joblib.dump(topic_model, os.path.join(MODEL_DIR, "topic_model.pkl"))
    joblib.dump(topic_vectorizer, os.path.join(MODEL_DIR, "topic_vectorizer.pkl"))

    nlp_metrics = {
        "sentiment_metrics": {
            "accuracy": round(sent_acc * 100, 2),
            "precision": round(sent_prec * 100, 2),
            "recall": round(sent_rec * 100, 2),
            "f1_score": round(sent_f1 * 100, 2),
            "algorithm": "Multinomial Naïve Bayes (TF-IDF)"
        },
        "topic_metrics": {
            "accuracy": round(topic_acc * 100, 2),
            "precision": round(topic_prec * 100, 2),
            "recall": round(topic_rec * 100, 2),
            "f1_score": round(topic_f1 * 100, 2),
            "algorithm": "Multinomial Naïve Bayes (TF-IDF)",
            "categories": sorted(list(y_topic.unique()))
        }
    }

    metrics_file = os.path.join(MODEL_DIR, "nlp_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(nlp_metrics, f, indent=4)

    print("Saved NLP artifacts:")
    print(" - sentiment_model.pkl")
    print(" - sentiment_vectorizer.pkl")
    print(" - topic_model.pkl")
    print(" - topic_vectorizer.pkl")
    print(" - nlp_metrics.json")
    print("\nNLP pipeline training & evaluation completed successfully!")

if __name__ == "__main__":
    main()
