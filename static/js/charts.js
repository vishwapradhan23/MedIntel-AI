/*
=============================================================================
MLIS University Project: AI-Driven Healthcare Prediction & Feedback System
Script: static/js/charts.js
Description: Chart.js initialization for Model Comparison, Dashboard Analytics,
             and Confusion Matrix rendering.
=============================================================================
*/

// Function to render Model Comparison Bar Chart (Module 2)
function renderComparisonChart(canvasId, metricsData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const models = Object.keys(metricsData);
    const accuracies = models.map(m => metricsData[m].accuracy);
    const precisions = models.map(m => metricsData[m].precision);
    const recalls = models.map(m => metricsData[m].recall);
    const f1s = models.map(m => metricsData[m].f1_score);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracies,
                    backgroundColor: 'rgba(26, 111, 176, 0.85)',
                    borderColor: '#1a6fb0',
                    borderWidth: 1
                },
                {
                    label: 'Precision (%)',
                    data: precisions,
                    backgroundColor: 'rgba(32, 178, 170, 0.85)',
                    borderColor: '#20b2aa',
                    borderWidth: 1
                },
                {
                    label: 'Recall (%)',
                    data: recalls,
                    backgroundColor: 'rgba(237, 108, 2, 0.85)',
                    borderColor: '#ed6c02',
                    borderWidth: 1
                },
                {
                    label: 'F1-Score (%)',
                    data: f1s,
                    backgroundColor: 'rgba(46, 125, 50, 0.85)',
                    borderColor: '#2e7d32',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 50,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

// Function to render Prediction Distribution Pie Chart (Module 6)
function renderPredictionPieChart(canvasId, diabeticCount, nonDiabeticCount) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['High Risk (Diabetic)', 'Low Risk (Non-Diabetic)'],
            datasets: [{
                data: [diabeticCount, nonDiabeticCount],
                backgroundColor: ['#d32f2f', '#2e7d32'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Function to render Sentiment Distribution Doughnut Chart (Module 6)
function renderSentimentDoughnutChart(canvasId, posCount, negCount, neuCount) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [posCount, negCount, neuCount],
                backgroundColor: ['#2e7d32', '#d32f2f', '#ed6c02'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}
