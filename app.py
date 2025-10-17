"""
Flask Web Application for ASIA SCI Outcome Prediction
Provides web interface for motor score and impairment grade predictions
"""

from flask import Flask, render_template, request, jsonify
from predict_motor_score import MotorScorePredictor
from predict_impairment_grade import ImpairmentGradePredictor
import os

app = Flask(__name__)

# Initialize predictors once at startup
print("Loading prediction models...")
motor_predictor = MotorScorePredictor()
grade_predictor = ImpairmentGradePredictor()
print("✓ Models loaded and ready!")

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get patient data from form
        patient_data = {
            'AInjAge': int(request.form.get('age', 0)),
            'ASex': int(request.form.get('sex', 1)),
            'ARace': int(request.form.get('race', 1)),
            'AHispnic': int(request.form.get('hispanic', 0)),
            'AMarStIj': int(request.form.get('marital', 1)),
            'AEducLvl': int(request.form.get('education', 3)),
            'APrLvlSt': int(request.form.get('language', 1)),
            'AFmIncLv': int(request.form.get('income', 3)),
            'APrimPay': int(request.form.get('payer', 1)),
            'APResInj': int(request.form.get('residence', 1)),
            'ADiabete': int(request.form.get('diabetes', 0)),
            'ADepress': int(request.form.get('depression', 0)),
            'AAnxiety': int(request.form.get('anxiety', 0)),
            'AAlcRate': int(request.form.get('alcohol_rate', 0)),
            'AAlcNbDr': int(request.form.get('alcohol_drinks', 0)),
            'AAlc6Mor': int(request.form.get('binge', 0)),
            'AI2RhADa': int(request.form.get('days_to_rehab', 25)),
            'ATrmEtio': int(request.form.get('etiology', 1)),
            'AAsscInj': int(request.form.get('associated_injuries', 0)),
            'AVertInj': int(request.form.get('vertebral_injury', 1)),
            'ASpinSrg': int(request.form.get('surgery', 0)),
            'AUMVAdm': int(request.form.get('ventilation', 0)),
            'AFScorRb': int(request.form.get('functional_score', 50)),
            'AASATotA': int(request.form.get('admission_motor', 50)),
            'AASAImAd': request.form.get('admission_grade', 'C').upper(),
            'ANurLvlA': request.form.get('neuro_level', 'T4').upper()
        }
        
        # Get predictions from both models
        motor_result = motor_predictor.predict(patient_data)
        grade_result = grade_predictor.predict(patient_data)
        
        # Prepare response
        response = {
            'success': True,
            'motor_prediction': {
                'predicted_score': motor_result['predicted_discharge_motor_score'],
                'admission_score': motor_result['admission_motor_score'],
                'expected_improvement': motor_result['expected_improvement'],
                'interpretation': get_motor_interpretation(motor_result['expected_improvement'])
            },
            'grade_prediction': {
                'predicted_grade': grade_result['predicted_grade'],
                'admission_grade': grade_result['admission_grade'],
                'confidence': grade_result['confidence'],
                'description': grade_result['predicted_grade_description'],
                'probabilities': grade_result['class_probabilities'],
                'interpretation': get_grade_interpretation(
                    grade_result['admission_grade'], 
                    grade_result['predicted_grade']
                )
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_motor_interpretation(improvement):
    """Get interpretation text for motor score improvement"""
    if improvement >= 20:
        return "EXCELLENT expected recovery (≥20 points)"
    elif improvement >= 10:
        return "GOOD expected recovery (10-20 points)"
    elif improvement >= 5:
        return "MODERATE expected recovery (5-10 points)"
    elif improvement >= 0:
        return "MINIMAL expected recovery (<5 points)"
    else:
        return "Possible decline (negative change)"

def get_grade_interpretation(admission_grade, predicted_grade):
    """Get interpretation text for grade prediction"""
    grade_order = ['A', 'B', 'C', 'D', 'E']
    
    if admission_grade == predicted_grade:
        return f"Grade expected to remain {predicted_grade} (stable)"
    elif admission_grade in grade_order and predicted_grade in grade_order:
        adm_idx = grade_order.index(admission_grade)
        pred_idx = grade_order.index(predicted_grade)
        if pred_idx > adm_idx:
            improvement = pred_idx - adm_idx
            return f"IMPROVEMENT expected: {admission_grade} → {predicted_grade} ({improvement} grade{'s' if improvement > 1 else ''})"
        else:
            return f"Decline expected: {admission_grade} → {predicted_grade}"
    else:
        return f"Change from {admission_grade} to {predicted_grade}"

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')

if __name__ == '__main__':
    # Run the app
    # Use port 5001 instead of 5000 (5000 is used by macOS AirPlay)
    port = int(os.environ.get('PORT', 5001))
    print(f"\n{'='*70}")
    print(f"🚀 ASIA SCI Prediction Server Starting...")
    print(f"{'='*70}")
    print(f"\n✓ Server running at: http://localhost:{port}")
    print(f"✓ Press Ctrl+C to stop the server")
    print(f"\n{'='*70}\n")
    app.run(host='0.0.0.0', port=port, debug=False)

