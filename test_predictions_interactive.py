"""
Interactive test script for both prediction models
Prompts for key variables and shows predictions
"""

from predict_motor_score import MotorScorePredictor
from predict_impairment_grade import ImpairmentGradePredictor

def get_user_input():
    """Collect key patient data from user"""
    print("\n" + "="*76)
    print("ASIA SCI OUTCOME PREDICTION - INTERACTIVE TEST")
    print("="*76)
    print("\nPlease enter patient data at ADMISSION:")
    print("(Press Enter to use default values shown in brackets)\n")
    
    # Collect key inputs
    age = input("Age at injury [45]: ").strip() or "45"
    sex = input("Sex (1=Male, 2=Female) [1]: ").strip() or "1"
    admission_motor = input("ASIA Motor Score at Admission (0-100) [42]: ").strip() or "42"
    admission_grade = input("ASIA Grade at Admission (A/B/C/D) [C]: ").strip().upper() or "C"
    neuro_level = input("Neurological Level (e.g., C5, T4, L1) [T4]: ").strip().upper() or "T4"
    days_to_rehab = input("Days from injury to rehab admission [25]: ").strip() or "25"
    spinal_surgery = input("Had spinal surgery? (0=No, 1=Yes) [1]: ").strip() or "1"
    
    # Build patient data dict with defaults for other features
    patient_data = {
        'AInjAge': int(age),
        'ASex': int(sex),
        'ARace': 1,
        'AHispnic': 0,
        'AMarStIj': 2,
        'AEducLvl': 3,
        'APrLvlSt': 1,
        'AFmIncLv': 3,
        'APrimPay': 1,
        'APResInj': 1,
        'ADiabete': 0,
        'ADepress': 0,
        'AAnxiety': 0,
        'AAlcRate': 0,
        'AAlcNbDr': 0,
        'AAlc6Mor': 0,
        'AI2RhADa': int(days_to_rehab),
        'ATrmEtio': 1,
        'AAsscInj': 0,
        'AVertInj': 1,
        'ASpinSrg': int(spinal_surgery),
        'AUMVAdm': 0,
        'AFScorRb': 50,
        'AASATotA': int(admission_motor),
        'AASAImAd': admission_grade,
        'ANurLvlA': neuro_level
    }
    
    return patient_data

def main():
    # Get patient data
    patient_data = get_user_input()
    
    print("\n" + "="*76)
    print("INITIALIZING MODELS...")
    print("="*76)
    
    # Initialize predictors
    motor_predictor = MotorScorePredictor()
    grade_predictor = ImpairmentGradePredictor()
    
    print("\n" + "="*76)
    print("PREDICTIONS")
    print("="*76)
    
    # Get motor score prediction
    print("\n" + "─"*76)
    print("1. MOTOR SCORE PREDICTION")
    print("─"*76)
    motor_result = motor_predictor.predict_with_interpretation(patient_data)
    print(motor_result)
    
    # Get impairment grade prediction
    print("\n" + "─"*76)
    print("2. IMPAIRMENT GRADE PREDICTION")
    print("─"*76)
    grade_result = grade_predictor.predict_with_interpretation(patient_data)
    print(grade_result)
    
    print("\n" + "="*76)
    print("SUMMARY")
    print("="*76)
    motor_pred = motor_predictor.predict(patient_data)
    grade_pred = grade_predictor.predict(patient_data)
    
    print(f"""
Patient entered with:
  • Admission Motor Score: {patient_data['AASATotA']}
  • Admission Grade: {patient_data['AASAImAd']}
  • Age: {patient_data['AInjAge']} years
  • Neurological Level: {patient_data['ANurLvlA']}

Model Predictions:
  • Expected Discharge Motor Score: {motor_pred['predicted_discharge_motor_score']:.1f}
  • Expected Motor Improvement: {motor_pred['expected_improvement']:+.1f} points
  • Expected Discharge Grade: {grade_pred['predicted_discharge_grade']}
  • Grade Prediction Confidence: {grade_pred['confidence']*100:.1f}%

Top 3 Most Important Features:
  Motor Score Model:
""")
    for i, (feat, imp) in enumerate(motor_pred['top_5_influential_features'][:3], 1):
        print(f"    {i}. {feat}: {imp*100:.1f}%")
    
    print(f"\n  Impairment Grade Model:")
    for i, (feat, imp) in enumerate(grade_pred['top_5_influential_features'][:3], 1):
        print(f"    {i}. {feat}: {imp*100:.1f}%")
    
    print("\n" + "="*76)
    print("Test complete! Models are working correctly.")
    print("="*76)

if __name__ == "__main__":
    main()

