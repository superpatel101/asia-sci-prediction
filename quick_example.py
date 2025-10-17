"""
Quick Example: Using the Random Forest Model for ASIA Motor Score Prediction

This is a simple example showing how to:
1. Load the trained model
2. Make predictions on new patient data
3. Interpret the results
"""

import joblib
import pandas as pd
import numpy as np

print("="*70)
print("QUICK START: ASIA Motor Score Prediction Model")
print("="*70)

# Step 1: Load the trained model and preprocessing artifacts
print("\n[Step 1] Loading model...")
model = joblib.load('random_forest_asia_motor_model.pkl')
imputer = joblib.load('imputer.pkl')
feature_names = joblib.load('feature_names.pkl')
print("✓ Model loaded successfully!")

# Step 2: Create example patient data
print("\n[Step 2] Creating example patient data...")
# Example: 3 hypothetical patients
example_patients = pd.DataFrame({
    # Demographics
    'AInjAge': [28, 45, 32],
    'ASex': [1, 2, 1],  # 1=Male, 2=Female
    'ARace': [1, 1, 2],
    'AHispnic': [0, 0, 0],
    'AMarStIj': [3, 2, 1],
    'AEducLvl': [2, 3, 3],
    'APrLvlSt': [1, 1, 1],
    'AFmIncLv': [9, 9, 9],
    'APrimPay': [4, 1, 3],
    'APResInj': [99, 99, 99],
    'APResDis': [6, 1, 1],
    
    # Medical History
    'ADiabete': [9, 9, 0],
    'ADepress': [9, 9, 9],
    'AAnxiety': [9, 9, 9],
    'AAlcRate': [9, 9, 9],
    'AAlcNbDr': [9, 9, 9],
    'AAlc6Mor': [9, 9, 9],
    'AI2RhADa': [9, 7, 10],
    
    # Injury Details
    'ATrmEtio': [30, 1, 1],
    'AAsscInj': [9, 9, 9],
    'AVertInj': [9, 9, 9],
    'ASpinSrg': [1, 1, 0],
    
    # Clinical Measures
    'AUMVAdm': [0, 0, 0],
    'AUMVDis': [0, 0, 0],
    'ABdMMDis': [5, 10, 13],
    'AFScorRb': [99, 99, 99],
    'AFScorDs': [99, 99, 99],
    'AASATotA': [11.0, 50.0, 40.0],  # ASIA total at admission
    'AASAImAd': [1, 5, 2],  # ASIA impairment at admission
    'AASAImDs': [1, 5, 3],  # ASIA impairment at discharge
    'ANurLvlA': ['C05', 'T12', 'C06'],  # Neurological level
    'ANurLvlD': ['C06', 'T12', 'C06']
})

print(f"✓ Created data for {len(example_patients)} patients")

# Step 3: Preprocess the data
print("\n[Step 3] Preprocessing data...")
# Handle categorical encoding
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['AInjAge', 'AASAImAd', 'AASAImDs', 'ANurLvlA', 'ANurLvlD']
for col in categorical_columns:
    if col in example_patients.columns and example_patients[col].dtype == 'object':
        le = LabelEncoder()
        example_patients[col] = le.fit_transform(example_patients[col].astype(str))

# Apply imputation
data_processed = pd.DataFrame(
    imputer.transform(example_patients[feature_names]), 
    columns=feature_names
)
print("✓ Data preprocessed")

# Step 4: Make predictions
print("\n[Step 4] Making predictions...")
predictions = model.predict(data_processed)
print("✓ Predictions complete!")

# Step 5: Display results
print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

for i, pred in enumerate(predictions, 1):
    admission_score = example_patients.loc[i-1, 'AASATotA']
    print(f"\nPatient {i}:")
    print(f"  - Admission ASIA Total Score: {admission_score:.1f}")
    print(f"  - Predicted Discharge Score:  {pred:.1f}")
    print(f"  - Expected Change:            {pred - admission_score:+.1f} points")
    
    # Clinical interpretation
    if pred >= 75:
        interpretation = "Excellent recovery expected"
    elif pred >= 50:
        interpretation = "Good recovery expected"
    elif pred >= 25:
        interpretation = "Moderate recovery expected"
    else:
        interpretation = "Limited recovery expected"
    print(f"  - Interpretation:             {interpretation}")

# Step 6: Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Mean predicted score:  {predictions.mean():.2f}")
print(f"Std predicted score:   {predictions.std():.2f}")
print(f"Min predicted score:   {predictions.min():.2f}")
print(f"Max predicted score:   {predictions.max():.2f}")

print("\n" + "="*70)
print("✓ Example complete!")
print("="*70)

# Additional tips
print("\n💡 TIPS:")
print("   • Admission scores (AASATotA) are the strongest predictor")
print("   • The model achieves 90.5% R² score on test data")
print("   • Average prediction error is ±8.3 points (RMSE)")
print("   • All 32 features are used for optimal predictions")
print("\n📖 For more details, see README.md and model_summary_report.txt\n")

