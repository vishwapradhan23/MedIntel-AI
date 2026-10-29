# Data Directory

The local SQLite database file (`healthcare.db`), chatbot conversational intents (`intents.json`), and triage datasets reside in this directory.

## Symptom Triage Datasets
- **`triage_train.csv`**: Contains synthetic training records (~1,600 samples, ~200 per specialty across 8 medical specialties) generated via `generate_triage_data.py` using varied clinical sentence templates, durations, informal phrasing, and negated symptoms.
- **`triage_test.csv`**: Contains hand-crafted, realistic patient symptom descriptions (≥168 samples, ≥21 per specialty) written with colloquial phrasing, spelling quirks, and varied sentence lengths. Kept strictly separate from the synthetic training set.

> **Important Clinical & Academic Note**:  
> The training data is synthetically synthesized from structured clinical vocabularies and the evaluation test set consists of hand-written patient vignettes. Consequently, all classification performance metrics (accuracy, precision, recall, F1) are **indicative decision-support evaluations only** and must not be construed as clinical diagnostic guarantees or medical trial outcomes.
