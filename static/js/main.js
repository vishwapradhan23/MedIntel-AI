/*
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: static/js/main.js
Description: General frontend interactivity, AJAX form processing,
             sample value populators, and user alerts.
=============================================================================
*/

document.addEventListener('DOMContentLoaded', () => {
    // Highlight Active Navbar Link
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // Auto-scroll chat history to bottom if on chatbot page
    const chatHistory = document.getElementById('chatHistory');
    if (chatHistory) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});

// Helper for displaying alert banners dynamically
function showAlert(containerId, message, type = 'warning') {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = `
        <div class="alert alert-${type}">
            <span>${message}</span>
        </div>
    `;
}

// Sample value filler for Diabetes Prediction Form
function fillSampleData(type) {
    if (type === 'diabetic') {
        document.getElementById('pregnancies').value = 5;
        document.getElementById('glucose').value = 166;
        document.getElementById('blood_pressure').value = 72;
        document.getElementById('skin_thickness').value = 25;
        document.getElementById('insulin').value = 175;
        document.getElementById('bmi').value = 35.8;
        document.getElementById('diabetes_pedigree').value = 0.587;
        document.getElementById('age').value = 51;
    } else if (type === 'normal') {
        document.getElementById('pregnancies').value = 1;
        document.getElementById('glucose').value = 85;
        document.getElementById('blood_pressure').value = 66;
        document.getElementById('skin_thickness').value = 23;
        document.getElementById('insulin').value = 80;
        document.getElementById('bmi').value = 22.6;
        document.getElementById('diabetes_pedigree').value = 0.251;
        document.getElementById('age').value = 26;
    }
}

// Sample text filler for Sentiment Analysis
function fillSampleFeedback(text) {
    const textElem = document.getElementById('feedback_text');
    if (textElem) {
        textElem.value = text;
    }
}

// Quick Prompt Sender for Chatbot
function sendPrompt(text) {
    const inputElem = document.getElementById('chatInput');
    if (inputElem) {
        inputElem.value = text;
        const form = document.getElementById('chatForm');
        if (form) {
            form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
    }
}
