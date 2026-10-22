# System Architecture & Academic Diagrams

This document contains complete diagrammatic representations matching the exact implementation of the **AI-Driven Healthcare Prediction & Patient Feedback Analysis System**.

---

## 1. System Architecture Diagram

```mermaid
graph TD
    User(["User / Patient"]) <--> WebUI["Web Interface (HTML5 / CSS3 / Vanilla JS)"]
    WebUI <--> FlaskApp["Flask Backend (app.py)"]
    
    subgraph Machine Learning Pipeline
        FlaskApp -->|Structured Input| Scaler["StandardScaler"]
        Scaler -->|Scaled Vector| KNN["KNN Model (knn_model.pkl)"]
        Scaler -->|Scaled Vector| NB["Naïve Bayes Model (naive_bayes_model.pkl)"]
        KNN -->|Prediction & Probabilities| PredOutput["Disease Prediction Result"]
        NB -->|Prediction & Probabilities| PredOutput
    end

    subgraph Natural Language Processing Pipeline
        FlaskApp -->|Feedback Text| TFIDF["TF-IDF Vectorizer (sentiment_vectorizer.pkl)"]
        TFIDF -->|TF-IDF Sparse Matrix| MNB_Sent["Multinomial NB (sentiment_model.pkl)"]
        TFIDF -->|TF-IDF Sparse Matrix| MNB_Topic["Multinomial NB (topic_model.pkl)"]
        MNB_Sent -->|Sentiment Label & Confidence| TextOutput["Feedback Analysis Result"]
        MNB_Topic -->|Domain Topic Category| TextOutput
    end

    subgraph Intelligent Local Chatbot Engine
        FlaskApp -->|User Query| ChatEngine["HealthcareChatbot (chatbot.py)"]
        ChatEngine -->|Pattern TF-IDF Matrix| CosineSim["Cosine Similarity Matching"]
        CosineSim -->|Confidence Check >= 15%| IntentResp["Intent Response + Disclaimer"]
    end

    PredOutput -->|Insert Record| DB[("SQLite Database (healthcare.db)")]
    TextOutput -->|Insert Record| DB
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

## 3. NLP Flowchart (Sentiment & Topic Classification)

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

## 4. Chatbot Flowchart (Local Intent Engine)

```mermaid
flowchart TD
    Start([Start]) --> LoadIntents["Load intents.json (13 Intent Tags & Patterns)"]
    LoadIntents --> VectorizePatterns["Build TF-IDF Matrix for all Pattern Sentences"]
    VectorizePatterns --> UserQuery[/User Enters Chatbot Question/]
    UserQuery --> CleanQuery["Normalize & Transform Query to TF-IDF Vector"]
    CleanQuery --> CosineMatch["Compute Cosine Similarity against all Intent Vectors"]
    CosineMatch --> FindMax["Find Max Similarity Score & Matched Intent Tag"]
    FindMax --> ThresholdCheck{"Is Max Score >= 0.15?"}
    ThresholdCheck -- Yes --> SelectResponse["Select Random Response for Matched Intent Tag"]
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

## 5. Use Case Diagram

```mermaid
gantt
    title Use Case Interaction Diagram
    dateFormat  X
    axisFormat %s

    section User Actions
    Disease Risk Prediction       :active, u1, 0, 1
    Model Comparison Overview    :active, u2, 1, 2
    Feedback Sentiment Analysis   :active, u3, 2, 3
    Interactive Chatbot Query     :active, u4, 3, 4
    View Dashboard & Log History :active, u5, 4, 5

    section System Responses
    ML Scaling & Inference        :active, s1, 0, 1
    Render Metrics & Bar Chart    :active, s2, 1, 2
    TF-IDF Vectorization & MNB    :active, s3, 2, 3
    Cosine Similarity Intent      :active, s4, 3, 4
    SQLite Database Query         :active, s5, 4, 5
```

---

## 6. Data Flow Diagram Level 0 (Context Diagram)

```mermaid
flowchart LR
    User(["User / Patient"]) -->|Health Parameters / Text / Query| System[["Healthcare AI System (app.py)"]]
    System -->|Prediction Results / Sentiment / Chatbot Response / Stats| User
```

---

## 7. Data Flow Diagram Level 1

```mermaid
flowchart TD
    User(["User"]) -->|1. Health Parameters| P1["1.0 Disease Risk Predictor"]
    P1 -->|Scale & Predict| M1[("ML Models: KNN / Naïve Bayes")]
    M1 -->|Prediction & Confidence| P1
    P1 -->|Store Record| D1[("SQLite Table: predictions")]
    P1 -->|Prediction Output| User

    User -->|2. Feedback Text| P2["2.0 Sentiment & Topic Classifier"]
    P2 -->|Vectorize & Classify| M2[("NLP Models: TF-IDF + MNB")]
    M2 -->|Sentiment & Topic Labels| P2
    P2 -->|Store Record| D2[("SQLite Table: feedback")]
    P2 -->|Sentiment Output| User

    User -->|3. Chat Query| P3["3.0 Healthcare Chatbot Engine"]
    P3 -->|Cosine Similarity| M3[("Intents Base: intents.json")]
    M3 -->|Intent Match & Response| P3
    P3 -->|Store Log| D3[("SQLite Table: chatbot_logs")]
    P3 -->|Bot Response| User

    D1 -->|Fetch Predictions| P4["4.0 Dashboard & History Controller"]
    D2 -->|Fetch Feedback| P4
    P4 -->|Render Charts & Tables| User
```

---

## 8. Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    PREDICTIONS {
        INTEGER id PK
        TEXT timestamp
        INTEGER pregnancies
        REAL glucose
        REAL blood_pressure
        REAL skin_thickness
        REAL insulin
        REAL bmi
        REAL diabetes_pedigree
        INTEGER age
        TEXT prediction
        TEXT model_used
        REAL confidence
        TEXT interpretation
    }

    FEEDBACK {
        INTEGER id PK
        TEXT timestamp
        TEXT feedback_text
        TEXT sentiment
        REAL sentiment_confidence
        TEXT topic
        REAL topic_confidence
    }

    CHATBOT_LOGS {
        INTEGER id PK
        TEXT timestamp
        TEXT user_query
        TEXT matched_intent
        REAL confidence
        TEXT bot_response
    }
```
