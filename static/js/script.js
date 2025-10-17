// Form submission and results handling

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('resultsSection');
    const submitBtn = form.querySelector('.submit-btn');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        submitBtn.classList.add('loading');
        submitBtn.textContent = '🔮 Predicting';
        submitBtn.disabled = true;
        
        // Get form data
        const formData = new FormData(form);
        
        try {
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
                resultsSection.style.display = 'block';
                
                // Scroll to results on mobile
                if (window.innerWidth < 1200) {
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error making prediction: ' + error.message);
            console.error('Prediction error:', error);
        } finally {
            // Reset button
            submitBtn.classList.remove('loading');
            submitBtn.textContent = '🔮 Predict Discharge Outcomes';
            submitBtn.disabled = false;
        }
    });
});

function displayResults(data) {
    // Motor Score Results
    const motorScore = data.motor_prediction.predicted_score.toFixed(1);
    const admissionScore = data.motor_prediction.admission_score;
    const motorChange = data.motor_prediction.expected_improvement.toFixed(1);
    
    document.getElementById('predictedMotorScore').textContent = motorScore;
    document.getElementById('admissionMotorScore').textContent = admissionScore;
    
    const changeElement = document.getElementById('motorChange');
    changeElement.textContent = (motorChange >= 0 ? '+' : '') + motorChange;
    changeElement.classList.toggle('negative', motorChange < 0);
    
    document.getElementById('motorInterpretation').textContent = data.motor_prediction.interpretation;
    
    // Grade Results
    document.getElementById('predictedGrade').textContent = 'Grade ' + data.grade_prediction.predicted_grade;
    document.getElementById('admissionGrade').textContent = 'Grade ' + data.grade_prediction.admission_grade;
    document.getElementById('gradeConfidence').textContent = data.grade_prediction.confidence;
    document.getElementById('gradeDescription').textContent = data.grade_prediction.description;
    document.getElementById('gradeInterpretation').textContent = data.grade_prediction.interpretation;
    
    // Probability bars
    displayProbabilityBars(data.grade_prediction.probabilities, data.grade_prediction.predicted_grade);
}

function displayProbabilityBars(probabilities, predictedGrade) {
    const container = document.getElementById('probabilityBars');
    container.innerHTML = '';
    
    // Grade descriptions
    const gradeDescriptions = {
        'A': 'Complete',
        'B': 'Sensory Incomplete',
        'C': 'Motor <50%',
        'D': 'Motor ≥50%',
        'E': 'Normal'
    };
    
    // Sort grades by probability (descending)
    const sortedGrades = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedGrades.forEach(([grade, prob]) => {
        const isPredicted = grade === predictedGrade;
        const probBarDiv = document.createElement('div');
        probBarDiv.className = 'prob-bar' + (isPredicted ? ' predicted' : '');
        
        probBarDiv.innerHTML = `
            <div class="prob-label">
                <span><strong>Grade ${grade}</strong> ${gradeDescriptions[grade] || ''} ${isPredicted ? '← PREDICTED' : ''}</span>
                <span><strong>${prob}%</strong></span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="width: ${prob}%">
                    ${prob > 15 ? prob + '%' : ''}
                </div>
            </div>
        `;
        
        container.appendChild(probBarDiv);
    });
}

function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    const icon = event.currentTarget.querySelector('.toggle-icon');
    
    if (section.style.display === 'none') {
        section.style.display = 'block';
        icon.textContent = '▲';
    } else {
        section.style.display = 'none';
        icon.textContent = '▼';
    }
}

// Auto-save form data to localStorage (optional - for user convenience)
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    
    // Load saved data
    const savedData = localStorage.getItem('asiaPredictionForm');
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            Object.keys(data).forEach(key => {
                const input = form.elements[key];
                if (input && data[key] !== undefined) {
                    input.value = data[key];
                }
            });
        } catch (e) {
            console.error('Error loading saved form data:', e);
        }
    }
    
    // Save data on change (debounced)
    let saveTimeout;
    form.addEventListener('change', function() {
        clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => {
            const formData = new FormData(form);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = value;
            }
            localStorage.setItem('asiaPredictionForm', JSON.stringify(data));
        }, 500);
    });
});

