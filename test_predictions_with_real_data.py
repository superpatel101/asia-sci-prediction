"""
Test predictions with ACTUAL data from training dataset
Verify predictions follow the trends we identified
"""

import pandas as pd
import numpy as np
from predict_motor_score_fixed import MotorScorePredictor

print("="*80)
print("TESTING PREDICTIONS WITH REAL TRAINING DATA")
print("="*80)

# Load the motor dataset
df = pd.read_csv('V2_EDIT_modelreadyASIAMotor.csv')
print(f"\n✓ Loaded dataset: {df.shape[0]} patients")

# Filter for patients with both admission and discharge scores
df_test = df[['AASATotA', 'AASATotD', 'AASAImAd', 'ANurLvlA', 'AInjAge', 'AI2RhADa', 
              'ASex', 'ARace', 'AHispnic', 'AMarStIj', 'AEducLvl', 'APrLvlSt', 'AFmIncLv',
              'APrimPay', 'APResInj', 'ADiabete', 'ADepress', 'AAnxiety', 'AAlcRate',
              'AAlcNbDr', 'AAlc6Mor', 'ATrmEtio', 'AAsscInj', 'AVertInj', 'ASpinSrg',
              'AUMVAdm', 'AFScorRb']].copy()

# Remove rows with missing admission or discharge scores
df_test = df_test.dropna(subset=['AASATotA', 'AASATotD'])

print(f"✓ {df_test.shape[0]} patients have complete admission & discharge scores")

# Initialize predictor
predictor = MotorScorePredictor()

print("\n" + "="*80)
print("TEST 1: HIGH ADMISSION SCORE (85) + GRADE D")
print("="*80)
print("Expected: Should predict IMPROVEMENT (not decline)")
print("Rationale: Grade D patients average +12.3 points improvement\n")

# Find a patient with high admission score and grade D
high_score_D = df_test[
    (df_test['AASATotA'] >= 80) & 
    (df_test['AASAImAd'].isin(['D', '4', '4.0']))
].head(3)

if len(high_score_D) > 0:
    for idx, row in high_score_D.iterrows():
        patient_data = row.to_dict()
        result = predictor.predict(patient_data)
        actual_discharge = row['AASATotD']
        actual_change = actual_discharge - row['AASATotA']
        
        print(f"Patient {idx}:")
        print(f"  Admission:  {row['AASATotA']:.0f} (Grade {row['AASAImAd']})")
        print(f"  ACTUAL Discharge: {actual_discharge:.0f} (Change: {actual_change:+.0f})")
        print(f"  PREDICTED Discharge: {result['predicted_discharge_motor_score']:.1f} (Change: {result['expected_improvement']:+.1f})")
        print(f"  ✓ Trend match: {'YES' if result['expected_improvement'] > 0 else 'NO - ERROR!'}")
        print()
else:
    print("No Grade D patients with admission score ≥80 found\n")

print("="*80)
print("TEST 2: GRADE C PATIENTS (BEST RECOVERY)")
print("="*80)
print("Expected: Should predict GOOD improvement (+25.6 average)")
print("Rationale: Grade C shows best motor recovery potential\n")

grade_C = df_test[
    (df_test['AASAImAd'].isin(['C', '3', '3.0'])) &
    (df_test['AASATotA'] >= 30) &
    (df_test['AASATotA'] <= 50)
].sample(n=min(5, len(df_test)), random_state=42)

errors = []
for idx, row in grade_C.iterrows():
    patient_data = row.to_dict()
    result = predictor.predict(patient_data)
    actual_discharge = row['AASATotD']
    actual_change = actual_discharge - row['AASATotA']
    
    print(f"Patient {idx}:")
    print(f"  Admission:  {row['AASATotA']:.0f} (Grade C)")
    print(f"  ACTUAL Discharge: {actual_discharge:.0f} (Change: {actual_change:+.0f})")
    print(f"  PREDICTED Discharge: {result['predicted_discharge_motor_score']:.1f} (Change: {result['expected_improvement']:+.1f})")
    error = abs(actual_discharge - result['predicted_discharge_motor_score'])
    errors.append(error)
    print(f"  Prediction error: {error:.1f} points")
    print()

print(f"Average prediction error for Grade C: {np.mean(errors):.2f} points")

print("="*80)
print("TEST 3: GRADE A PATIENTS (MINIMAL RECOVERY)")
print("="*80)
print("Expected: Should predict MINIMAL improvement (+3.8 average)")
print("Rationale: Grade A (complete injury) has limited recovery\n")

grade_A = df_test[
    (df_test['AASAImAd'].isin(['A', '1', '1.0'])) &
    (df_test['AASATotA'] <= 20)
].sample(n=min(5, len(df_test)), random_state=42)

errors_A = []
for idx, row in grade_A.iterrows():
    patient_data = row.to_dict()
    result = predictor.predict(patient_data)
    actual_discharge = row['AASATotD']
    actual_change = actual_discharge - row['AASATotA']
    
    print(f"Patient {idx}:")
    print(f"  Admission:  {row['AASATotA']:.0f} (Grade A)")
    print(f"  ACTUAL Discharge: {actual_discharge:.0f} (Change: {actual_change:+.0f})")
    print(f"  PREDICTED Discharge: {result['predicted_discharge_motor_score']:.1f} (Change: {result['expected_improvement']:+.1f})")
    error = abs(actual_discharge - result['predicted_discharge_motor_score'])
    errors_A.append(error)
    print(f"  Prediction error: {error:.1f} points")
    print()

print(f"Average prediction error for Grade A: {np.mean(errors_A):.2f} points")

print("="*80)
print("TEST 4: USER'S EXAMPLE (85 admission, Grade D)")
print("="*80)
print("Testing the exact scenario the user reported as incorrect\n")

user_example = {
    'AInjAge': 45,
    'ASex': 1,
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
    'AI2RhADa': 25,
    'ATrmEtio': 1,
    'AAsscInj': 0,
    'AVertInj': 1,
    'ASpinSrg': 1,
    'AUMVAdm': 0,
    'AFScorRb': 50,
    'AASATotA': 85,      # High admission score
    'AASAImAd': 'D',     # Grade D
    'ANurLvlA': 'T4'
}

result = predictor.predict(user_example)
print(f"User's example:")
print(f"  Admission:  85 (Grade D)")
print(f"  PREDICTED Discharge: {result['predicted_discharge_motor_score']:.1f}")
print(f"  PREDICTED Change: {result['expected_improvement']:+.1f} points")
print(f"\n  ✓ Expected: POSITIVE change (Grade D avg +12.3 points)")
print(f"  ✓ Actual prediction: {'POSITIVE ✓' if result['expected_improvement'] > 0 else 'NEGATIVE ✗ ERROR!'}")

print("\n" + "="*80)
print("TEST 5: COMPARE ACTUAL VS PREDICTED DISTRIBUTIONS")
print("="*80)

# Sample random patients
sample = df_test.sample(n=min(100, len(df_test)), random_state=42)
predictions = []
actuals = []

for idx, row in sample.iterrows():
    patient_data = row.to_dict()
    result = predictor.predict(patient_data)
    predictions.append(result['predicted_discharge_motor_score'])
    actuals.append(row['AASATotD'])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate overall metrics
mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals)**2))
correlation = np.corrcoef(predictions, actuals)[0, 1]

print(f"\n100-patient sample statistics:")
print(f"  MAE: {mae:.2f} points")
print(f"  RMSE: {rmse:.2f} points")
print(f"  Correlation: {correlation:.3f}")
print(f"\n  Actual mean discharge: {actuals.mean():.1f}")
print(f"  Predicted mean discharge: {predictions.mean():.1f}")

# Check if predictions are reasonable
reasonable_count = np.sum((predictions >= 0) & (predictions <= 100))
print(f"\n  Predictions in valid range (0-100): {reasonable_count}/100")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nIf predictions follow our trends:")
print("  ✓ Grade D (high admission) → positive improvement")
print("  ✓ Grade C → good improvement (~20-25 points)")
print("  ✓ Grade A → minimal improvement (~3-5 points)")
print("  ✓ Predictions should be within ±12 points (RMSE) on average")
print("  ✓ All predictions should be 0-100 range")

