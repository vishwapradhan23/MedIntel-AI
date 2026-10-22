"""
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: chatbot.py
Description: Local Intelligent Healthcare Chatbot engine implementing
             TF-IDF Vector Space Model and Cosine Similarity Intent Matching.
=============================================================================
"""

import os
import re
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS_FILE = r"c:\Users\Vishwa Pradhan\OneDrive\Desktop\MedIntel AI\data\intents.json"

class HealthcareChatbot:
    def __init__(self, intents_path=INTENTS_FILE):
        self.intents_path = intents_path
        self.intents = []
        self.patterns = []
        self.pattern_intent_map = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self.load_intents()
        self.build_model()

    def preprocess(self, text):
        """Clean and normalize text query."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def load_intents(self):
        """Load intent patterns and responses from JSON knowledge base."""
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents knowledge base missing at {self.intents_path}")

        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.intents = data.get("intents", [])

        for intent in self.intents:
            tag = intent["tag"]
            for pattern in intent.get("patterns", []):
                cleaned_p = self.preprocess(pattern)
                if cleaned_p:
                    self.patterns.append(cleaned_p)
                    self.pattern_intent_map.append(tag)

    def build_model(self):
        """Build TF-IDF matrix for all intent pattern sentences."""
        if not self.patterns:
            print("Warning: No intent patterns loaded.")
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def get_response(self, user_message):
        """
        Process user query and return intent tag, matched response, and confidence score.
        """
        cleaned_input = self.preprocess(user_message)
        
        if not cleaned_input or len(cleaned_input) < 2:
            fallback = self.get_intent_by_tag("unknown")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "response": random.choice(fallback["responses"])
            }

        # Transform user query into TF-IDF vector space
        user_vector = self.vectorizer.transform([cleaned_input])

        # Calculate Cosine Similarity against all intent patterns
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        max_score = float(similarities[best_idx])

        # Confidence Threshold: If similarity is low, trigger fallback
        CONFIDENCE_THRESHOLD = 0.15

        if max_score >= CONFIDENCE_THRESHOLD:
            matched_intent_tag = self.pattern_intent_map[best_idx]
            confidence_pct = round(max_score * 100, 1)
        else:
            matched_intent_tag = "unknown"
            confidence_pct = 0.0

        intent_obj = self.get_intent_by_tag(matched_intent_tag)
        selected_response = random.choice(intent_obj["responses"])

        return {
            "intent": matched_intent_tag,
            "confidence": confidence_pct,
            "response": selected_response
        }

    def get_intent_by_tag(self, tag):
        """Retrieve intent object by its tag name."""
        for intent in self.intents:
            if intent["tag"] == tag:
                return intent
        # Default fallback
        return {
            "tag": "unknown",
            "responses": ["I am an educational assistant. Please ask about diabetes symptoms, diet, prevention, or using this portal."]
        }

# Instantiate singleton instance for import
chatbot_engine = HealthcareChatbot()

if __name__ == "__main__":
    print("Testing Healthcare Chatbot Engine locally...")
    test_queries = [
        "Hello there!",
        "What are the symptoms of diabetes?",
        "How can I prevent high blood sugar?",
        "Tell me about healthy diet and exercise",
        "How does the prediction page work?",
        "What is the capital of France?"  # Out of domain query
    ]

    for q in test_queries:
        res = chatbot_engine.get_response(q)
        print(f"\nUser Query : '{q}'")
        print(f"Matched Intent: {res['intent']} (Confidence: {res['confidence']}%)")
        print(f"Chatbot Output: {res['response']}")
