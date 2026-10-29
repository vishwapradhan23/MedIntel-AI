"""
=============================================================================
MedIntel AI: Symptom Triage & Red-Flag Clinical Decision Engine
Script: triage_engine.py
Description: Evaluates patient free-text symptoms, identifies clinical Red Flags,
             determines triage urgency level (EMERGENCY / SEE SOON / ROUTINE),
             and infers top-3 medical specialty probabilities using trained
             TF-IDF + Logistic Regression classifiers.
=============================================================================
"""

import os
import re
import joblib
import numpy as np

# Model Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "triage_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "triage_vectorizer.pkl")

# Global in-memory model instances
_triage_model = None
_triage_vectorizer = None

def load_triage_artifacts():
    """Load or reload serialized triage model and vectorizer into memory."""
    global _triage_model, _triage_vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            _triage_model = joblib.load(MODEL_PATH)
            _triage_vectorizer = joblib.load(VECTORIZER_PATH)
            return True
        except Exception as e:
            print(f"Error loading triage model artifacts: {e}")
            return False
    return False

# Initialize on module load
load_triage_artifacts()

# =============================================================================
# CLINICAL RED-FLAG DEFINITIONS
# =============================================================================

# Emergency patterns indicating immediate life-threat (EMERGENCY)
EMERGENCY_RULES = [
    {
        "category": "Cardiac / Acute Coronary Syndrome",
        "regex": r"(chest pain|chest pressure|chest tightness|squeezing.*chest|heavy.*chest).*(left arm|radiat.*arm|jaw|neck|back|shoulder|sweat|breathless)",
        "explanation": "Symptoms suggestive of acute coronary syndrome (possible myocardial infarction / heart attack)."
    },
    {
        "category": "Severe Crushing Chest Pain",
        "regex": r"(crushing chest|elephant.*chest|severe chest pain|acute chest pain|chest pain.*(sweating|dizziness|faint))",
        "explanation": "Severe central chest pain requiring urgent cardiac assessment."
    },
    {
        "category": "Stroke FAST Signs (Neurological Deficit)",
        "regex": r"(facial droop|face.*droop|drooping.*face|slurr.*speech|difficulty speak|cannot speak|weakness.*one side|one-sided weakness|paraly.*(arm|leg|side)|hemiparesis)",
        "explanation": "Sudden neurological deficit consistent with acute cerebrovascular accident (Stroke FAST signs)."
    },
    {
        "category": "Thunderclap Headache",
        "regex": r"(worst headache of.*life|thunderclap headache|sudden explosive headache|headache.*loss of consciousness)",
        "explanation": "Sudden onset catastrophic headache suspicious for subarachnoid hemorrhage."
    },
    {
        "category": "Severe Respiratory Failure",
        "regex": r"(cannot breathe|gasping for air|suffocating|struggling to breathe|lips.*blue|turning blue|severe stridor|choking)",
        "explanation": "Severe acute respiratory distress or compromised airway."
    },
    {
        "category": "Hemoptysis (Coughing Blood)",
        "regex": r"(coughing up blood|blood in cough|hemoptysis)",
        "explanation": "Frank blood in sputum indicating potential pulmonary vascular or airway compromise."
    },
    {
        "category": "Upper GI Bleed / Hematemesis",
        "regex": r"(vomiting blood|threw up blood|blood in vomit|coffee ground vomit|hematemesis|black tarry stool)",
        "explanation": "Signs of active acute gastrointestinal hemorrhage."
    },
    {
        "category": "Anaphylaxis / Severe Allergy",
        "regex": r"(throat closing|tongue.*swell|swelling.*(throat|tongue)|difficulty swallow.*breathing|anaphylaxis)",
        "explanation": "Signs of acute systemic anaphylaxis with potential airway occlusion."
    },
    {
        "category": "Syncope / Sudden Unconsciousness",
        "regex": r"(loss of consciousness|passed out|blacked out|unresponsive|collapsed suddenly)",
        "explanation": "Transient or persistent loss of consciousness requiring urgent diagnostic stabilization."
    },
    {
        "category": "Meningismus / Sepsis Signs",
        "regex": r"(stiff neck.*(high fever|fever.*rash)|fever.*stiff neck|rash.*not fade|petechial rash)",
        "explanation": "Suspicion of acute bacterial meningitis or systemic meningococcal infection."
    }
]

# Urgent patterns requiring prompt medical evaluation (SEE SOON: within 24-48 hours)
SEE_SOON_RULES = [
    {
        "category": "Prolonged High Pyrexia",
        "regex": r"(high fever.*(3 days|4 days|5 days|week)|fever.*103|fever.*104|temperature.*103|temperature.*104)",
        "explanation": "Prolonged high-grade fever unresponsive to simple antipyretics."
    },
    {
        "category": "Acute Abdominal Pain (Appendicitis / Cholecystitis)",
        "regex": r"(lower right.*abdomen|right lower.*belly|sharp.*belly pain.*fever|severe stomach pain.*vomit)",
        "explanation": "Localized severe abdominal pain warranting timely surgical or medical evaluation."
    },
    {
        "category": "Dehydration from Persistent Emesis",
        "regex": r"(cannot keep (water|fluids) down|vomiting.*(all day|24 hours|two days)|severe dehydration)",
        "explanation": "Persistent inability to maintain oral hydration."
    },
    {
        "category": "Acute Vision Change",
        "regex": r"(sudden (blindness|vision loss|double vision|blurred vision.*headache)|aura.*migraine)",
        "explanation": "Acute changes in visual acuity or new ocular symptoms."
    },
    {
        "category": "Acute Inability to Bear Weight",
        "regex": r"(cannot bear weight|inability to walk.*joint|joint.*swollen.*red.*warm|severe knee swelling)",
        "explanation": "Acute musculoskeletal trauma or possible septic/inflammatory arthritis."
    },
    {
        "category": "Diabetic Hyperglycemia Alert",
        "regex": r"(blood sugar.*(250|300|350|400)|glucose.*(250|300|350|400)|extreme thirst.*rapid weight loss)",
        "explanation": "Significant metabolic dysregulation suggesting uncontrolled diabetes or ketosis."
    }
]

DISCLAIMER = (
    "CLINICAL DECISION SUPPORT NOTICE: MedIntel AI Symptom Triage is an artificial intelligence decision-support "
    "tool designed for educational and informational purposes only. It DOES NOT provide a formal medical diagnosis. "
    "If you or someone around you is experiencing life-threatening symptoms, chest pain, stroke indicators, acute "
    "shortness of breath, or serious trauma, call your national emergency medical services (such as 911, 112, or 999) "
    "or proceed to the nearest emergency hospital immediately."
)

def evaluate_red_flags(symptom_text):
    """
    Scan raw patient text for clinical emergency and urgent red-flag patterns.
    Returns:
        urgency: 'EMERGENCY' | 'SEE SOON' | 'ROUTINE'
        red_flags: list of dicts with category and explanation
        action_advice: clinical action recommendation string
    """
    text_lower = symptom_text.lower()
    matched_flags = []

    # 1. Check EMERGENCY rules first
    for rule in EMERGENCY_RULES:
        if re.search(rule["regex"], text_lower):
            matched_flags.append({
                "severity": "EMERGENCY",
                "category": rule["category"],
                "explanation": rule["explanation"]
            })

    if matched_flags:
        return {
            "urgency": "EMERGENCY",
            "red_flags": matched_flags,
            "action_advice": "CRITICAL: Proceed immediately to the nearest Emergency Department (ED) or call emergency medical services."
        }

    # 2. Check SEE SOON rules second
    for rule in SEE_SOON_RULES:
        if re.search(rule["regex"], text_lower):
            matched_flags.append({
                "severity": "SEE SOON",
                "category": rule["category"],
                "explanation": rule["explanation"]
            })

    if matched_flags:
        return {
            "urgency": "SEE SOON",
            "red_flags": matched_flags,
            "action_advice": "ATTENTION: Schedule an in-person clinical consultation with a physician within 24 to 48 hours."
        }

    # 3. Default to ROUTINE
    return {
        "urgency": "ROUTINE",
        "red_flags": [],
        "action_advice": "Standard medical consultation: Schedule an elective appointment with the recommended medical specialist or your primary care physician."
    }

def clean_triage_text(text):
    """Normalize text while retaining negation words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"won['’]t", "wont", text)
    text = re.sub(r"can['’]t", "cant", text)
    text = re.sub(r"n['’]t", "nt", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def predict_triage(symptom_text):
    """
    End-to-End Triage Evaluation:
    1. Preprocesses text (preserving negation tokens)
    2. Runs red-flag clinical urgency rules
    3. Infers primary medical specialty and top-3 probabilities
    4. Formulates explainable summary and medical disclaimer
    """
    if not symptom_text or len(symptom_text.strip()) < 3:
        raise ValueError("Symptom description must be at least 3 characters long.")

    cleaned_text = clean_triage_text(symptom_text)

    # 1. Red-Flag Evaluation
    red_flag_result = evaluate_red_flags(symptom_text)
    urgency = red_flag_result["urgency"]
    red_flags = red_flag_result["red_flags"]
    action_advice = red_flag_result["action_advice"]

    # 2. Specialty Inference
    global _triage_model, _triage_vectorizer
    if _triage_model is None or _triage_vectorizer is None:
        load_triage_artifacts()

    top_3_specialties = []
    primary_specialty = "General Physician"
    primary_confidence = 70.0

    if _triage_model is not None and _triage_vectorizer is not None:
        try:
            vec = _triage_vectorizer.transform([cleaned_text])
            probabilities = _triage_model.predict_proba(vec)[0]
            classes = _triage_model.classes_

            # Sort classes by descending probability
            sorted_indices = np.argsort(probabilities)[::-1]
            top_classes = sorted_indices[:3]

            for idx in top_classes:
                spec_name = str(classes[idx])
                pct = round(float(probabilities[idx]) * 100, 1)
                top_3_specialties.append({
                    "specialty": spec_name,
                    "probability": pct
                })

            primary_specialty = top_3_specialties[0]["specialty"]
            primary_confidence = top_3_specialties[0]["probability"]
        except Exception as e:
            print(f"Error during triage model inference: {e}")
            top_3_specialties = fallback_specialty_heuristics(cleaned_text)
            primary_specialty = top_3_specialties[0]["specialty"]
            primary_confidence = top_3_specialties[0]["probability"]
    else:
        # Robust heuristic fallback if model not yet saved
        top_3_specialties = fallback_specialty_heuristics(cleaned_text)
        primary_specialty = top_3_specialties[0]["specialty"]
        primary_confidence = top_3_specialties[0]["probability"]

    # 3. Extract key terms for explainability
    words = [w for w in cleaned_text.split() if len(w) > 3][:6]

    return {
        "success": True,
        "raw_input": symptom_text,
        "cleaned_input": cleaned_text,
        "suggested_specialty": primary_specialty,
        "confidence": primary_confidence,
        "top_3": top_3_specialties,
        "urgency": urgency,
        "red_flags": red_flags,
        "action_advice": action_advice,
        "salient_terms": words,
        "disclaimer": DISCLAIMER
    }

def fallback_specialty_heuristics(cleaned_text):
    """Clinical keyword heuristic fallback when offline models are not loaded."""
    scores = {
        "Cardiology": 0, "Neurology": 0, "Gastroenterology": 0,
        "Dermatology": 0, "Orthopedics": 0, "Pulmonology": 0,
        "Endocrinology": 0, "General Physician": 1  # default baseline
    }

    # Keyword mappings
    if any(w in cleaned_text for w in ["chest", "heart", "palpitation", "pulse", "angina", "arm pain"]):
        scores["Cardiology"] += 4
    if any(w in cleaned_text for w in ["headache", "migraine", "dizzy", "vertigo", "tingling", "numb", "tremor", "seizure"]):
        scores["Neurology"] += 4
    if any(w in cleaned_text for w in ["stomach", "cramp", "reflux", "heartburn", "vomit", "diarrhea", "constipation", "bloat", "nausea", "belly"]):
        scores["Gastroenterology"] += 4
    if any(w in cleaned_text for w in ["rash", "itch", "skin", "eczema", "hives", "mole", "acne", "scales", "flaky", "peel"]):
        scores["Dermatology"] += 4
    if any(w in cleaned_text for w in ["knee", "joint", "back", "spine", "ankle", "sprain", "shoulder", "bone", "stiff", "fracture", "hip"]):
        scores["Orthopedics"] += 4
    if any(w in cleaned_text for w in ["cough", "wheez", "breath", "lung", "phlegm", "asthma", "bronchial", "inhaler"]):
        scores["Pulmonology"] += 4
    if any(w in cleaned_text for w in ["thirst", "urination", "weight loss", "weight gain", "thyroid", "cold intolerant", "glucose", "sugar", "sweat"]):
        scores["Endocrinology"] += 4
    if any(w in cleaned_text for w in ["fever", "chills", "fatigue", "body ache", "flu", "weakness", "tired"]):
        scores["General Physician"] += 3

    total = sum(scores.values()) or 1
    sorted_specs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    result = []
    for spec, score in sorted_specs:
        prob = round(max((score / total) * 100, 10.0), 1)
        result.append({"specialty": spec, "probability": prob})
    return result
