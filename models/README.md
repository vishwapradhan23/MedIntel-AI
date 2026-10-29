# Models Directory

This directory stores serialized binary model artifacts (`.pkl` via Joblib) and evaluation metrics (`.json`) for all MedIntel AI intelligent modules:

## Serialized Machine Learning & NLP Models
- **`knn_model.pkl`**: Fitted K-Nearest Neighbors estimator ($k=7$, distance weights) for diabetes risk prediction.
- **`naive_bayes_model.pkl`**: Fitted Gaussian Naïve Bayes estimator for metabolic risk classification.
- **`logistic_regression_model.pkl`**: Fitted Logistic Regression baseline for diabetes comparison.
- **`scaler.pkl`**: Fitted `StandardScaler` containing mean and variance parameters for 8 clinical biometrics.
- **`sentiment_model.pkl`**: Fitted Multinomial Naïve Bayes classifier for 3-class patient feedback sentiment analysis.
- **`sentiment_vectorizer.pkl`**: Fitted `TfidfVectorizer` (unigrams + bigrams) for review sentiment tokenization.
- **`topic_model.pkl`**: Fitted Multinomial Naïve Bayes classifier for 7 operational healthcare topic domains.
- **`topic_vectorizer.pkl`**: Fitted `TfidfVectorizer` (unigrams + bigrams) for operational feedback categorization.
- **`triage_model.pkl`**: Fitted `LogisticRegression` (class_weight='balanced') classifier for 8-class symptom specialty triage.
- **`triage_vectorizer.pkl`**: Fitted `TfidfVectorizer` (unigrams + bigrams, sublinear_tf=True) preserving negation tokens.

## Evaluation & Benchmark Metrics
- **`disease_metrics.json`**: Empirical test metrics (Accuracy, Precision, Recall, F1, Confusion Matrices) comparing KNN, GaussianNB, and Logistic Regression.
- **`nlp_metrics.json`**: Empirical performance metrics for Patient Feedback Sentiment and Operational Topic classifiers.
- **`triage_metrics.json`**: Rigorous evaluation metrics on held-out hand-written test data comparing the Primary Triage Model against baselines (MultinomialNB, Unigram-only TF-IDF, Sentence-Transformers) along with confusion matrices and top predictive clinical terms.
