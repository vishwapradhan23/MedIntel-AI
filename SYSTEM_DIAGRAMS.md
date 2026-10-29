# System Architecture & Academic Diagrams

This document contains complete diagrammatic representations matching the exact implementation of the **MedIntel AI: AI-Driven Healthcare Prediction, Feedback Analysis & Symptom Triage Platform**.

---

## 1. System Architecture Diagram

```mermaid
graph TD
    User(["User / Clinician / Patient"]) <--> WebUI["Modern Web Interface (HTML5 / CSS3 Glassmorphism / Vanilla JS)"]
    WebUI <--> FlaskApp["Flask Backend Service (app.py)"]
    
    subgraph Structured Biometric Pipeline
        FlaskApp -->|Structured Biometrics| Scaler["StandardScaler Z-Score Normalization"]
        Scaler -->|Scaled Vector| KNN["KNN Classifier (knn_model.pkl)"]
        Scaler -->|Scaled Vector| NB["Gaussian Naïve Bayes (naive_bayes_model.pkl)"]
        KNN -->|Prediction & Probabilities| PredOutput["Disease Risk Assessment"]
        NB -->|Prediction & Probabilities| PredOutput
    end

    subgraph Unstructured Review NLP Pipeline
        FlaskApp -->|Feedback Text| TFIDF_Sent["TF-IDF Vectorizer (sentiment_vectorizer.pkl)"]
        TFIDF_Sent -->|TF-IDF Sparse Matrix| MNB_Sent["Multinomial NB (sentiment_model.pkl)"]
        TFIDF_Sent -->|TF-IDF Sparse Matrix| MNB_Topic["Multinomial NB (topic_model.pkl)"]
        MNB_Sent -->|Sentiment Label & Confidence| TextOutput["Feedback Analysis Result"]
        MNB_Topic -->|Domain Topic Category| TextOutput
    end

    subgraph Symptom Triage & Red-Flag Pipeline
        FlaskApp -->|Free-Text Symptoms| TriageEngine["Triage Engine (triage_engine.py)"]
        TriageEngine -->|Regex Analysis| RedFlagCheck{"Clinical Red Flags?"}
        RedFlagCheck -- Emergency Match --> UrgencyEmerg["Urgency: EMERGENCY (Pulse Red Alert)"]
        RedFlagCheck -- Urgent Match --> UrgencySoon["Urgency: SEE SOON (Amber Alert)"]
        RedFlagCheck -- No Match --> UrgencyRoutine["Urgency: ROUTINE (Green Standard)"]
        TriageEngine -->|Preserve Negations| TFIDF_Triage["TF-IDF (1,2) (triage_vectorizer.pkl)"]
        TFIDF_Triage --> LogReg_Triage["Logistic Regression (triage_model.pkl)"]
        LogReg_Triage --> Top3Specialties["Top-3 Medical Specialty Probabilities"]
    end

    subgraph Intelligent Local Chatbot Engine
        FlaskApp -->|User Query| ChatEngine["HealthcareChatbot (chatbot.py)"]
        ChatEngine -->|Pattern TF-IDF Matrix| CosineSim["Cosine Similarity Matching"]
        CosineSim -->|Confidence Check >= 15%| IntentResp["Intent Response + Educational Notice"]
    end

    PredOutput -->|Insert Record| DB[("SQLite Database (healthcare.db)")]
    TextOutput -->|Insert Record| DB
    UrgencyEmerg --> DB
    UrgencySoon --> DB
    UrgencyRoutine --> DB
    Top3Specialties --> DB
    IntentResp -->|Log Query| DB

    DB -->|Fetch Analytics & History| WebUI
```

---

## 2. Machine Learning Flowchart (Disease Prediction)

```mermaid
flowchart TD
    Start([Start]) --> LoadData["Load Diabetes Dataset (diabetes.csv)"]
    LoadData --> CleanZero["Impute Zero Values in Glucose, BP, BMI with Medians"]
    CleanZero --> SplitData["Train-Test Split (80% Train / 20% Test)"]
    SplitData --> FitScaler["Fit StandardScaler on X_train"]
    FitScaler --> TrainKNN["Train KNN Classifier (k=7, distance weights)"]
    FitScaler --> TrainNB["Train Gaussian Naïve Bayes Classifier"]
    TrainKNN --> EvalModels["Evaluate Test Accuracy, Precision, Recall, F1, Confusion Matrix"]
    TrainNB --> EvalModels
    EvalModels --> SaveArtifacts["Save Serialized Models (.pkl) & disease_metrics.json"]
    
    SaveArtifacts --> WebInput[/User Enters Clinical Parameters on Web Form/]
    WebInput --> ValidateInput{"Are Inputs Valid & Non-Negative?"}
    ValidateInput -- No --> ShowError[Display Validation Error]
    ValidateInput -- Yes --> TransformInput["Transform Input via StandardScaler"]
    TransformInput --> ModelChoice{"Model Choice?"}
    ModelChoice -- KNN --> PredictKNN["Predict via KNN Model"]
    ModelChoice -- Naïve Bayes --> PredictNB["Predict via Gaussian NB Model"]
    PredictKNN --> CalcProb["Calculate Confidence Probability"]
    PredictNB --> CalcProb
    CalcProb --> GenExplanation["Generate Clinical Interpretation & Disclaimer"]
    GenExplanation --> LogDB[(Save Record in SQLite Table: predictions)]
    LogDB --> RenderUI[/Display Result Badge & Explanation on Web UI/]
    RenderUI --> End([End])
```

---

## 3. Symptom Triage & Red-Flag Clinical Flowchart

```mermaid
flowchart TD
    Start([Start]) --> PatientInput[/Patient Enters Free-Text Symptoms/]
    PatientInput --> CleanText["Normalize Text (Preserve Negations: no, not, without, never, dont)"]
    
    CleanText --> RedFlagCheck{"Check Deterministic Red-Flag Rules"}
    RedFlagCheck -- Severe Cardiac / Stroke FAST / Bleed / Airway Compromise --> SetEmergency["Urgency = EMERGENCY; Trigger Pulsing Red Banner & Immediate 911/112 Advice"]
    RedFlagCheck -- High Fever >3 Days / Appendicitis / Severe Dehydration --> SetSoon["Urgency = SEE SOON; Trigger Amber Warning & 24-48h Consultation Advice"]
    RedFlagCheck -- Mild Chronic / Elective Symptoms --> SetRoutine["Urgency = ROUTINE; Trigger Green Standard Elective Appointment Advice"]
    
    CleanText --> VectorizeTriage["TF-IDF Vectorizer (Unigrams + Bigrams, Sublinear TF)"]
    VectorizeTriage --> ModelPredict["Logistic Regression (Balanced Class Weights)"]
    ModelPredict --> CalcProba["Compute Softmax Probabilities across 8 Specialties"]
    CalcProba --> RankTop3["Extract & Rank Top-3 Probabilities (%)"]
    
    SetEmergency --> MergeResult["Synthesize Triage Report"]
    SetSoon --> MergeResult
    SetRoutine --> MergeResult
    RankTop3 --> MergeResult
    
    MergeResult --> ExtractTerms["Extract Salient Terms for Clinical Explainability"]
    ExtractTerms --> AppendDisclaimer["Append Non-Diagnostic Decision Support Disclaimer"]
    AppendDisclaimer --> SaveSQL[(Persist Transaction in SQLite Table: triage_logs)]
    SaveSQL --> RenderTriageUI[/Display Urgency Badge, Top-3 Probability Bars & Red-Flag Warnings/]
    RenderTriageUI --> End([End])
```

---

## 4. NLP Flowchart (Feedback Sentiment & Topic Classification)

```mermaid
flowchart TD
    Start([Start]) --> LoadCorpora["Load sentiment.csv & topic_feedback.csv"]
    LoadCorpora --> Preprocess["Lowercasing & Regex Cleaning (Strip Digits/Punctuation)"]
    Preprocess --> TFIDF["Fit TfidfVectorizer (Unigrams & Bigrams)"]
    TFIDF --> TrainSent["Train Sentiment Classifier (Multinomial NB)"]
    TFIDF --> TrainTopic["Train Topic Classifier (Multinomial NB)"]
    TrainSent --> SaveNLP["Save NLP Models & nlp_metrics.json"]
    TrainTopic --> SaveNLP

    SaveNLP --> InputText[/User Enters Feedback Text on Web Form/]
    InputText --> CleanText["Normalize Text & Tokenize"]
    CleanText --> VectorizeText["Transform Text via TfidfVectorizer"]
    VectorizeText --> InferSent["Predict Sentiment (Positive / Negative / Neutral)"]
    VectorizeText --> InferTopic["Predict Topic (7 Domain Categories)"]
    InferSent --> ExtractTerms["Extract Top TF-IDF Key Terms"]
    InferTopic --> ExtractTerms
    ExtractTerms --> SaveSQL[(Save Record in SQLite Table: feedback)]
    SaveSQL --> DisplayResult[/Render Sentiment Badge, Topic Chip & Terms on UI/]
    DisplayResult --> End([End])
```

---

## 5. Chatbot Flowchart (Local Intent Engine)

```mermaid
flowchart TD
    Start([Start]) --> LoadIntents["Load intents.json (13 Intent Tags & Patterns)"]
    LoadIntents --> VectorizePatterns["Build TF-IDF Matrix for all Pattern Sentences"]
    VectorizePatterns --> UserQuery[/User Enters Chatbot Question/]
    UserQuery --> CleanQuery["Normalize & Transform Query to TF-IDF Vector"]
    CleanQuery --> CosineMatch["Compute Cosine Similarity against all Intent Vectors"]
    CosineMatch --> FindMax["Find Max Similarity Score & Matched Intent Tag"]
    FindMax --> ThresholdCheck{"Is Max Score >= 0.15?"}
    ThresholdCheck -- Yes --> SelectResponse["Select Response for Matched Intent Tag"]
    ThresholdCheck -- No --> SelectFallback["Select Unknown Fallback Response"]
    SelectResponse --> MedicalCheck{"Is Medical Intent?"}
    MedicalCheck -- Yes --> AppendDisclaimer["Append Medical Disclaimer"]
    MedicalCheck -- No --> FormatOutput["Format Final Bot Text"]
    SelectFallback --> FormatOutput
    AppendDisclaimer --> FormatOutput
    FormatOutput --> LogChat[(Log Entry in SQLite Table: chatbot_logs)]
    LogChat --> RenderBubble[/Render Chat Bubble & Confidence Score on UI/]
    RenderBubble --> End([End])
```

---

## 6. Data Flow Diagram Level 0 (Context Diagram)

```mermaid
flowchart LR
    User(["User / Patient / Clinician"]) -->|Health Biometrics / Symptoms / Feedback / Query| System[["MedIntel AI Platform (Flask app.py)"]]
    System -->|Risk Prediction / Triage Urgency & Specialties / Sentiment / Chat Responses / Analytics| User
```

---

## 7. Data Flow Diagram Level 1

```mermaid
flowchart TD
    User(["User"]) -->|1. Health Biometrics| P1["1.0 Disease Risk Predictor"]
    P1 -->|Scale & Predict| M1[("ML Models: KNN / Naïve Bayes")]
    M1 -->|Prediction & Confidence| P1
    P1 -->|Store Record| D1[("SQLite Table: predictions")]
    P1 -->|Prediction Output| User

    User -->|2. Free-Text Symptoms| P2["2.0 Symptom Triage & Red-Flag Engine"]
    P2 -->|Scan Red-Flags & Classify| M2[("Triage Engine: Regex + TF-IDF LogReg")]
    M2 -->|Urgency Level & Top-3 Probabilities| P2
    P2 -->|Store Record| D2[("SQLite Table: triage_logs")]
    P2 -->|Triage Output & Action Advice| User

    User -->|3. Feedback Text| P3["3.0 Sentiment & Topic Classifier"]
    P3 -->|Vectorize & Classify| M3[("NLP Models: TF-IDF + MNB")]
    M3 -->|Sentiment & Topic Labels| P3
    P3 -->|Store Record| D3[("SQLite Table: feedback")]
    P3 -->|Sentiment Output| User

    User -->|4. Chat Query| P4["4.0 Healthcare Chatbot Engine"]
    P4 -->|Cosine Similarity| M4[("Intents Base: intents.json")]
    M4 -->|Intent Match & Response| P4
    P4 -->|Store Log| D4[("SQLite Table: chatbot_logs")]
    P4 -->|Bot Response| User

    D1 -->|Fetch Predictions| P5["5.0 Dashboard & History Controller"]
    D2 -->|Fetch Triage Logs| P5
    D3 -->|Fetch Feedback| P5
    P5 -->|Render Charts & Tables| User
```

---

## 8. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    PREDICTIONS {
        INTEGER id PK "Auto Increment"
        TEXT timestamp "YYYY-MM-DD HH:MM:SS"
        INTEGER pregnancies "Count"
        REAL glucose "Plasma glucose mg/dL"
        REAL blood_pressure "Diastolic mm Hg"
        REAL skin_thickness "Triceps mm"
        REAL insulin "2-hour serum muU/mL"
        REAL bmi "Body mass index kg/m2"
        REAL diabetes_pedigree "Genetic pedigree score"
        INTEGER age "Years"
        TEXT prediction "High Risk / Low Risk"
        TEXT model_used "KNN / Naïve Bayes"
        REAL confidence "Percentage 0-100"
        TEXT interpretation "Clinical interpretation text"
    }

    TRIAGE_LOGS {
        INTEGER id PK "Auto Increment"
        TEXT timestamp "YYYY-MM-DD HH:MM:SS"
        TEXT symptom_text "Patient free-text symptoms"
        TEXT predicted_specialty "Primary medical specialty"
        REAL confidence "Probability percentage"
        TEXT top_3_json "Top 3 specialties with probabilities JSON"
        TEXT urgency "EMERGENCY / SEE SOON / ROUTINE"
        TEXT red_flags_matched "Matched clinical red-flags"
    }

    FEEDBACK {
        INTEGER id PK "Auto Increment"
        TEXT timestamp "YYYY-MM-DD HH:MM:SS"
        TEXT feedback_text "Patient review string"
        TEXT sentiment "Positive / Negative / Neutral"
        REAL sentiment_confidence "Probability percentage"
        TEXT topic "Operational domain topic"
        REAL topic_confidence "Probability percentage"
    }

    CHATBOT_LOGS {
        INTEGER id PK "Auto Increment"
        TEXT timestamp "YYYY-MM-DD HH:MM:SS"
        TEXT user_query "Raw user input prompt"
        TEXT matched_intent "Detected intent tag"
        REAL confidence "Cosine similarity score"
        TEXT bot_response "Dispatched reply text"
    }
```
