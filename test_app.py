"""
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: test_app.py
Description: End-to-end automated test suite verifying ML inference, NLP pipelines,
             local Chatbot intent matching, SQLite persistence, and Flask REST API endpoints.
=============================================================================
"""

import os
import json
import unittest
from app import app, load_all_models
import database
from chatbot import chatbot_engine

class HealthcareAISystemTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment and initialize database."""
        load_all_models()
        database.init_db()
        database.seed_initial_data()
        cls.client = app.test_client()

    def test_01_health_prediction_api_valid(self):
        """Test Disease Prediction API with valid high-risk parameters."""
        payload = {
            "pregnancies": 5,
            "glucose": 166,
            "blood_pressure": 72,
            "skin_thickness": 25,
            "insulin": 175,
            "bmi": 35.8,
            "diabetes_pedigree": 0.587,
            "age": 51,
            "model_choice": "KNN"
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn("Diabetic", data['prediction'])
        self.assertGreater(data['confidence'], 50)
        self.assertIn("glucose", data['interpretation'].lower())

    def test_02_health_prediction_api_invalid_input(self):
        """Test Disease Prediction API with invalid negative parameters."""
        payload = {
            "pregnancies": -2,
            "glucose": -100,
            "blood_pressure": 70,
            "skin_thickness": 20,
            "insulin": 80,
            "bmi": 25.0,
            "diabetes_pedigree": 0.35,
            "age": 30,
            "model_choice": "KNN"
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])
        self.assertIn("cannot be negative", data['error'])

    def test_03_sentiment_and_topic_api(self):
        """Test Patient Feedback Sentiment & Topic Analysis API."""
        payload = {
            "feedback_text": "The doctor was extremely patient and explained my medical condition clearly."
        }
        response = self.client.post('/api/sentiment', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['sentiment'], "Positive")
        self.assertEqual(data['topic'], "Doctor/Consultation")
        self.assertGreater(data['sentiment_confidence'], 50)

    def test_04_chatbot_api_known_intent(self):
        """Test Local Chatbot API with a known healthcare query."""
        payload = {
            "message": "What are the common symptoms of diabetes?"
        }
        response = self.client.post('/api/chatbot', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['matched_intent'], "diabetes_symptoms")
        self.assertGreater(data['confidence'], 15)
        self.assertIn("urination", data['bot_response'].lower())

    def test_05_chatbot_api_fallback_intent(self):
        """Test Local Chatbot API with an out-of-domain query (triggers fallback)."""
        payload = {
            "message": "What is the airspeed velocity of an unladen swallow?"
        }
        response = self.client.post('/api/chatbot', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['matched_intent'], "unknown")
        self.assertEqual(data['confidence'], 0.0)

    def test_06_metrics_api(self):
        """Test Model Metrics API endpoint."""
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn("disease_metrics", data)

    def test_07_dashboard_stats_api(self):
        """Test Dashboard Statistics API endpoint."""
        response = self.client.get('/api/dashboard')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn("stats", data)
        self.assertGreaterEqual(data['stats']['total_predictions'], 0)

    def test_08_history_api(self):
        """Test History API endpoint."""
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn("predictions", data)
        self.assertIn("feedback", data)

if __name__ == '__main__':
    print("=" * 60)
    print("   RUNNING SYSTEM INTEGRATION & VALIDATION TEST SUITE     ")
    print("=" * 60)
    unittest.main()
