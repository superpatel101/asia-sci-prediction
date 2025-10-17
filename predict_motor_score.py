"""
ASIA Motor Score Prediction - FIXED Inference Script
Uses explicit categorical encoding that matches training
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

class MotorScorePredictor:
    def __init__(self):
        """Load the trained motor score model and preprocessing artifacts"""
        print("Loading ASIA Motor Score Prediction Model...")
        self.model = joblib.load('random_forest_motor_clean_model.pkl')
        self.imputer = joblib.load('motor_clean_imputer.pkl')
        self.feature_names = joblib.load('motor_clean_feature_names.pkl')
        print(f"✓ Model loaded successfully")
        print(f"✓ Requires {len(self.feature_names)} features")
        
    def encode_categorical_features(self, df):
        """Apply EXPLICIT categorical encoding (matches training)"""
        df = df.copy()
        
        # 1. AInjAge: Convert to numeric
        df['AInjAge'] = pd.to_numeric(df['AInjAge'], errors='coerce')
        
        # 2. AASAImAd: ASIA Impairment Grade mapping
        asia_grade_map = {
            '1': 1, '1.0': 1, 'A': 1,  # Complete
            '2': 2, '2.0': 2, 'B': 2,  # Sensory Incomplete
            '3': 3, '3.0': 3, 'C': 3,  # Motor Incomplete <50%
            '4': 4, '4.0': 4, 'D': 4,  # Motor Incomplete ≥50%
            '5': 5, '5.0': 5, 'E': 5,  # Normal
            '9': 9, '9.0': 9, 'U': 9   # Unknown
        }
        if 'AASAImAd' in df.columns:
            df['AASAImAd'] = df['AASAImAd'].astype(str).map(asia_grade_map)
        
        # 3. ANurLvlA: Neurological level - extract level number
        # Format: C01-C08 (1-8), T01-T12 (9-20), L01-L05 (21-25), S01-S05 (26-30)
        def encode_neuro_level(level):
            if pd.isna(level):
                return np.nan
            level_str = str(level).strip().upper()
            if len(level_str) < 2:
                return np.nan
            try:
                segment = level_str[0]
                number = int(level_str[1:])
                if segment == 'C':
                    return number  # 1-8
                elif segment == 'T':
                    return 8 + number  # 9-20
                elif segment == 'L':
                    return 20 + number  # 21-25
                elif segment == 'S':
                    return 25 + number  # 26-30
                else:
                    return np.nan
            except:
                return np.nan
        
        if 'ANurLvlA' in df.columns:
            df['ANurLvlA'] = df['ANurLvlA'].apply(encode_neuro_level)
        
        return df
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict discharge motor score from admission data
        """
        # Create DataFrame with patient data
        df = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Apply EXPLICIT categorical encoding (same as training)
        df = self.encode_categorical_features(df)
        
        # Apply imputation
        X = self.imputer.transform(df)
        
        # Make prediction
        predicted_score = self.model.predict(X)[0]
        
        # Get feature importances
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate expected improvement
        admission_score = patient_data.get('AASATotA', 0)
        expected_improvement = predicted_score - admission_score
        
        return {
            'predicted_discharge_motor_score': round(predicted_score, 1),
            'admission_motor_score': admission_score,
            'expected_improvement': round(expected_improvement, 1),
            'top_5_influential_features': top_features,
            'confidence_note': 'Prediction based on Random Forest model trained on 10,543 patients'
        }
    
    def predict_with_interpretation(self, patient_data: Dict[str, Any]) -> str:
        """
        Predict and return formatted interpretation
        """
        result = self.predict(patient_data)
        
        interpretation = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                 ASIA MOTOR SCORE PREDICTION                              ║
╚══════════════════════════════════════════════════════════════════════════╝

PATIENT DATA:
  • Admission Motor Score:  {result['admission_motor_score']:.1f} / 100
  • Admission Grade:         {patient_data.get('AASAImAd', 'Unknown')}
  • Age:                     {patient_data.get('AInjAge', 'Unknown')} years
  • Days to Rehab:           {patient_data.get('AI2RhADa', 'Unknown')} days

PREDICTION:
  ┌─────────────────────────────────────────────────────────────┐
  │  Predicted Discharge Motor Score: {result['predicted_discharge_motor_score']:.1f} / 100
  │  Expected Improvement:            {result['expected_improvement']:+.1f} points
  └─────────────────────────────────────────────────────────────┘

INTERPRETATION:
"""
        
        # Add interpretation based on improvement
        improvement = result['expected_improvement']
        if improvement >= 20:
            interpretation += "  ✓ EXCELLENT expected recovery (≥20 points)\n"
        elif improvement >= 10:
            interpretation += "  ✓ GOOD expected recovery (10-20 points)\n"
        elif improvement >= 5:
            interpretation += "  ○ MODERATE expected recovery (5-10 points)\n"
        elif improvement >= 0:
            interpretation += "  ○ MINIMAL expected recovery (<5 points)\n"
        else:
            interpretation += "  ⚠ Possible decline (negative change)\n"
        
        interpretation += f"\nTOP 5 MOST INFLUENTIAL FEATURES FOR THIS PREDICTION:\n"
        for i, (feature, importance) in enumerate(result['top_5_influential_features'], 1):
            interpretation += f"  {i}. {feature}: {importance*100:.1f}% importance\n"
        
        interpretation += f"\n{result['confidence_note']}\n"
        interpretation += "="*76 + "\n"
        
        return interpretation


if __name__ == "__main__":
    print("="*76)
    print("ASIA MOTOR SCORE PREDICTOR - INFERENCE MODE")
    print("="*76)
    print("This script loads the trained model and predicts discharge motor score")
    print("from admission patient data.")
    
    # Initialize predictor
    predictor = MotorScorePredictor()
    
    # Example patient data
    example_patient = {
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
        'AASATotA': 42,       # Admission motor score
        'AASAImAd': 'C',      # Admission grade: C
        'ANurLvlA': 'T4'      # Neurological level: T4
    }
    
    # Make prediction with interpretation
    print(predictor.predict_with_interpretation(example_patient))
    
    # Also show raw result
    result = predictor.predict(example_patient)
    print("\nRaw prediction result:")
    print(result)
    
    print("\n" + "="*76)
    print("TO USE IN YOUR APPLICATION:")
    print("="*76)
    print("""
from predict_motor_score_fixed import MotorScorePredictor

# Initialize once
predictor = MotorScorePredictor()

# Make predictions
result = predictor.predict(patient_data_dict)
print(f"Predicted discharge score: {result['predicted_discharge_motor_score']}")

# Or get formatted interpretation
interpretation = predictor.predict_with_interpretation(patient_data_dict)
print(interpretation)
    """)

