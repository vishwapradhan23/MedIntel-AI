# MedIntel AI
### AI-Driven Healthcare Prediction & Patient Feedback Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, explainable healthcare analytics platform combining supervised machine learning for disease risk prediction, natural language processing for patient feedback sentiment and operational topic classification, and a local privacy-preserving conversational assistant.

---

## Overview

MedIntel AI bridges structured physiological measurements and unstructured clinical text into a single, unified local web application. The platform operates completely offline without external cloud API dependencies, ensuring patient data confidentiality, zero API overhead, and mathematical transparency for clinical and academic evaluation.

### Core Capabilities

- **Disease Risk Prediction**: Predicts diabetes risk from 8 clinical features (Glucose, Insulin, BMI, Blood Pressure, Age, etc.) using K-Nearest Neighbors (KNN, $k=7$) and Gaussian Naïve Bayes classifiers with Z-score feature scaling.
- **Model Comparison Benchmarks**: Empirical side-by-side evaluation of KNN, Gaussian Naïve Bayes, and Logistic Regression on test partition metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrices).
- **Patient Feedback Sentiment Analysis**: Classifies patient reviews into Positive, Negative, or Neutral sentiment using TF-IDF feature extraction and a Multinomial Naïve Bayes classifier.
- **Operational Topic Classification**: Categorizes feedback into 7 hospital operational domains (*Doctor/Consultation, Treatment, Staff, Waiting Time, Facilities, Prediction/System, General Feedback*).
- **Local Healthcare Chatbot**: Lightweight conversational assistant powered by a TF-IDF Vector Space Model and Cosine Similarity geometric intent matching.
- **Real-Time Analytics Dashboards**: Interactive Chart.js visualizations for disease risk distributions, sentiment breakdowns, and comprehensive audit logs.

---

## System Architecture

```
User / Web Browser (HTML5 / Vanilla CSS3 / JavaScript)
                        │
                        ▼
            Flask REST API Backend (app.py)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Structured Biometric Pipeline     Unstructured Text Pipeline
       │                                 │
StandardScaler Z-Score            TF-IDF Vector Space Model
       │                                 │
KNN / Gaussian Naïve Bayes        Multinomial Naïve Bayes
       │                          (Sentiment & Topic Classifiers)
       ▼                                 │
Disease Risk Outcome                     ▼
       │                          Feedback Analytics Outcome
       │                                 │
       └────────────────┬────────────────┘
                        ▼
             Embedded Database Storage
                        │
                        ▼
          Real-Time Analytics Dashboard
```

---

## Technology Stack

- **Runtime & Backend**: Python 3.9+, Flask
- **Machine Learning & Data Science**: Scikit-Learn, Pandas, NumPy, Joblib
- **Natural Language Processing**: TF-IDF Vectorizer, Cosine Similarity
- **Database**: SQLite3 (ACID-compliant embedded relational database)
- **Frontend**: Semantic HTML5, Vanilla CSS3 (Custom Healthcare Design System), Vanilla JavaScript (ES6+)
- **Visualization**: Chart.js

---

## Project Structure

```
MedIntel-AI/
├── app.py                         # Main Flask application and REST endpoints
├── database.py                    # Database schema definition and query helpers
├── chatbot.py                     # Local TF-IDF Cosine Similarity chatbot engine
├── train_models.py                # Supervised ML model training pipeline
├── train_nlp.py                   # NLP TF-IDF and Naive Bayes training pipeline
├── test_app.py                    # Automated test suite
├── requirements.txt               # Pinned Python package dependencies
├── COMPLETE_PROJECT_REPORT.md     # Detailed academic and technical project report
├── README.md                      # Repository overview and setup guide
│
├── data/                          # Data storage
│   ├── healthcare.db              # SQLite database file
│   └── intents.json               # Chatbot intent patterns and medical responses
│
├── datasets/                      # Training datasets
│   ├── diabetes.csv               # Pima Indians Diabetes benchmark dataset
│   ├── sentiment.csv              # Patient review sentiment corpus
│   └── topic_feedback.csv         # Hospital operational feedback corpus
│
├── models/                        # Pre-trained models and serialized metrics
│   ├── knn_model.pkl              # Fitted K-Nearest Neighbors estimator
│   ├── naive_bayes_model.pkl      # Fitted Gaussian Naive Bayes estimator
│   ├── logistic_regression_model.pkl # Fitted Logistic Regression estimator
│   ├── scaler.pkl                 # Fitted StandardScaler normalization parameters
│   ├── sentiment_model.pkl        # Fitted MultinomialNB sentiment classifier
│   ├── sentiment_vectorizer.pkl   # Fitted TF-IDF vectorizer for sentiment
│   ├── topic_model.pkl            # Fitted MultinomialNB topic classifier
│   ├── topic_vectorizer.pkl       # Fitted TF-IDF vectorizer for topics
│   ├── disease_metrics.json       # Evaluated test metrics for disease models
│   └── nlp_metrics.json           # Evaluated test metrics for NLP models
│
├── static/                        # Frontend assets
│   ├── css/
│   │   └── style.css              # Custom responsive stylesheet
│   └── js/
│       ├── main.js                # Form helpers and sample data injectors
│       └── charts.js              # Chart.js initialization routines
│
└── templates/                     # Jinja2 HTML5 presentation templates
    ├── base.html                  # Master layout template
    ├── home.html                  # Platform dashboard and module launcher
    ├── predict.html               # Disease risk prediction interface
    ├── comparison.html            # Model comparison benchmarks and confusion matrices
    ├── sentiment.html             # Patient feedback NLP analysis interface
    ├── chatbot.html               # Conversational healthcare assistant interface
    ├── prediction_analytics.html  # Disease prediction statistics and logs
    └── feedback_analytics.html    # Feedback sentiment breakdown and audit logs
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MedIntel-AI.git
cd MedIntel-AI
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain Models or Re-initialize Database

Pre-trained models and seeded databases are included. To retrain from scratch:

```bash
# Retrain Disease Prediction Models (KNN, Naive Bayes, Logistic Regression)
python train_models.py

# Retrain NLP Models (Sentiment and Topic Classifiers)
python train_nlp.py

# Initialize or reset the database
python database.py
```

### 5. Launch the Web Application

```bash
python app.py
```

Open your browser and navigate to:
**http://127.0.0.1:5000**

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | `POST` | Accepts 8 clinical parameters and returns risk classification, confidence score, and interpretation. |
| `/api/sentiment` | `POST` | Accepts feedback text and returns sentiment polarity, confidence score, and domain topic category. |
| `/api/chatbot` | `POST` | Processes user query and returns matched intent, cosine similarity score, and medical response. |
| `/api/metrics` | `GET` | Returns accuracy, precision, recall, F1-scores, and confusion matrices for all trained models. |
| `/api/dashboard` | `GET` | Returns aggregated statistics including total predictions, high/low risk ratios, and sentiment percentages. |
| `/api/history` | `GET` | Returns recent prediction audit logs and patient feedback records. |

---

## Running Automated Tests

Run the test suite to verify model inference, API status codes, and input validation rules:

```bash
python -m unittest test_app.py
```

---

## Model Benchmark Summary

| Model | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|
| **K-Nearest Neighbors ($k=7$)** | **90.0%** | **100.0%** | **72.2%** | **83.9%** |
| **Logistic Regression** | 84.0% | 77.8% | 77.8% | 77.8% |
| **Gaussian Naïve Bayes** | 82.0% | 73.7% | 77.8% | 75.7% |

*Evaluated on the benchmark Pima Indians Diabetes test partition.*

---

## Disclaimers

### Academic Use Only
This application is an educational software project developed for the **Machine Learning for Intelligent Systems (MLIS)** curriculum.

1. **Not a Medical Device**: This platform is not certified as a diagnostic tool by any clinical or health regulatory body.
2. **Statistical Outputs**: All predictions and confidence scores represent statistical patterns from benchmark datasets and must never replace clinical laboratory tests or consultation with a certified medical professional.
3. **Emergency Situations**: In case of a medical emergency, immediately contact local emergency medical services.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
