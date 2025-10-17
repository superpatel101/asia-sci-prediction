"""
ASIA Impairment Grade Prediction - FIXED Inference Script  
Uses explicit categorical encoding that matches training
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

class ImpairmentGradePredictor:
    def __init__(self):
        """Load the trained impairment classifier and preprocessing artifacts"""
        print("Loading ASIA Impairment Grade Prediction Model...")
        self.model = joblib.load('random_forest_impairment_classifier.pkl')
        self.imputer = joblib.load('impairment_imputer.pkl')
        self.feature_names = joblib.load('impairment_feature_names.pkl')
        self.grade_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        self.grade_descriptions = {
            'A': 'Complete - No motor or sensory function below injury',
            'B': 'Sensory Incomplete - Sensory but no motor function below injury',
            'C': 'Motor Incomplete - Motor function present, <50% of key muscles have grade ≥3',
            'D': 'Motor Incomplete - Motor function present, ≥50% of key muscles have grade ≥3',
            'E': 'Normal - Motor and sensory function normal'
        }
        print(f"✓ Model loaded successfully")
        print(f"✓ Requires {len(self.feature_names)} features")
        print(f"✓ Predicts grades: {list(self.grade_map.values())}")
        
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
        Predict discharge impairment grade from admission data
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
        predicted_class = self.model.predict(X)[0]
        predicted_grade = self.grade_map[predicted_class]
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[0]
        class_probabilities = {
            self.grade_map[cls]: round(prob * 100, 1)
            for cls, prob in zip(self.model.classes_, probabilities)
        }
        
        # Get confidence (probability of predicted class)
        confidence = class_probabilities[predicted_grade]
        
        return {
            'predicted_grade': predicted_grade,
            'predicted_grade_description': self.grade_descriptions[predicted_grade],
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'admission_grade': patient_data.get('AASAImAd', 'Unknown')
        }
    
    def predict_with_interpretation(self, patient_data: Dict[str, Any]) -> str:
        """
        Predict and return formatted interpretation
        """
        result = self.predict(patient_data)
        
        interpretation = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              ASIA IMPAIRMENT GRADE PREDICTION                            ║
╚══════════════════════════════════════════════════════════════════════════╝

PATIENT DATA:
  • Admission Grade:         {result['admission_grade']}
  • Age:                     {patient_data.get('AInjAge', 'Unknown')} years
  • Neurological Level:      {patient_data.get('ANurLvlA', 'Unknown')}

PREDICTION:
  ┌─────────────────────────────────────────────────────────────┐
  │  Predicted Discharge Grade: {result['predicted_grade']}
  │  Confidence: {result['confidence']:.1f}%
  └─────────────────────────────────────────────────────────────┘

GRADE DESCRIPTION:
  {result['predicted_grade_description']}

PROBABILITY DISTRIBUTION:
"""
        
        for grade in ['A', 'B', 'C', 'D', 'E']:
            if grade in result['class_probabilities']:
                prob = result['class_probabilities'][grade]
                bar = '█' * int(prob / 5)
                interpretation += f"  Grade {grade}: {prob:5.1f}%  {bar}\n"
        
        interpretation += f"\nBased on Random Forest model trained on 15,053 patients\n"
        interpretation += "="*76 + "\n"
        
        return interpretation


if __name__ == "__main__":
    print("="*76)
    print("ASIA IMPAIRMENT GRADE PREDICTOR - INFERENCE MODE")
    print("="*76)
    
    # Initialize predictor
    predictor = ImpairmentGradePredictor()
    
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
from predict_impairment_grade_fixed import ImpairmentGradePredictor

# Initialize once
predictor = ImpairmentGradePredictor()

# Make predictions
result = predictor.predict(patient_data_dict)
print(f"Predicted grade: {result['predicted_grade']} ({result['confidence']:.1f}% confidence)")
    """)

