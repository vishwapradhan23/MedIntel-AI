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
        self.assertIn("triages", data)

    def test_09_triage_api_emergency_cardiac(self):
        """Test Symptom Triage API: Cardiac Red-Flag triggers EMERGENCY & Cardiology."""
        payload = {
            "symptoms": "Severe crushing chest pain since morning, spreading to my left arm with cold sweat"
        }
        response = self.client.post('/api/triage', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['urgency'], "EMERGENCY")
        self.assertEqual(data['suggested_specialty'], "Cardiology")
        self.assertGreater(len(data['red_flags']), 0)
        self.assertEqual(len(data['top_3']), 3)
        self.assertIn("disclaimer", data)

    def test_10_triage_api_emergency_neurology_fast(self):
        """Test Symptom Triage API: Stroke FAST Red-Flag triggers EMERGENCY & Neurology."""
        payload = {
            "symptoms": "Sudden facial droop and slurred speech when trying to talk"
        }
        response = self.client.post('/api/triage', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['urgency'], "EMERGENCY")
        self.assertEqual(data['suggested_specialty'], "Neurology")
        self.assertIn("Stroke", data['red_flags'][0]['category'])

    def test_11_triage_api_routine_dermatology(self):
        """Test Symptom Triage API: Non-urgent dermatological rash classifies as ROUTINE."""
        payload = {
            "symptoms": "Extremely itchy red rash and dry flaky patches on both elbows for the past two weeks"
        }
        response = self.client.post('/api/triage', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['urgency'], "ROUTINE")
        self.assertEqual(data['suggested_specialty'], "Dermatology")
        self.assertEqual(len(data['red_flags']), 0)

    def test_12_triage_api_negation_handling(self):
        """Test Symptom Triage API: Correctly parses negated symptoms without false specialty triggers."""
        payload = {
            "symptoms": "no fever or cough but severe sharp pain in lower right abdomen with vomiting"
        }
        response = self.client.post('/api/triage', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['suggested_specialty'], "Gastroenterology")
        self.assertIn(data['urgency'], ["SEE SOON", "EMERGENCY"])

    def test_13_triage_api_invalid_input(self):
        """Test Symptom Triage API: Rejects empty or too-short symptom descriptions."""
        payload = {"symptoms": "a"}
        response = self.client.post('/api/triage', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])
        self.assertIn("at least 3 characters", data['error'])

    def test_14_triage_metrics_endpoint(self):
        """Test Model Metrics API returns triage metrics."""
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn("triage_metrics", data)

    def test_15_triage_page_view(self):
        """Test GET /triage renders triage HTML page successfully."""
        response = self.client.get('/triage')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Symptom Triage with Red-Flag Detection", response.data)

if __name__ == '__main__':
    print("=" * 60)
    print("   RUNNING SYSTEM INTEGRATION & VALIDATION TEST SUITE     ")
    print("=" * 60)
    unittest.main()
