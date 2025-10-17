"""
ASIA Motor Score Prediction - Inference Script
Load trained model and predict discharge motor score from admission data
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
        
    def get_feature_info(self) -> Dict[str, str]:
        """Return information about required features"""
        feature_descriptions = {
            'AInjAge': 'Age at injury (years)',
            'ASex': 'Sex (1=Male, 2=Female, 3=Other, 9=Unknown)',
            'ARace': 'Race code',
            'AHispnic': 'Hispanic (0=No, 1=Yes, 9=Unknown)',
            'AMarStIj': 'Marital status at injury',
            'AEducLvl': 'Education level',
            'APrLvlSt': 'Primary language spoken',
            'AFmIncLv': 'Family income level',
            'APrimPay': 'Primary payer',
            'APResInj': 'Place of residence before injury',
            'ADiabete': 'Diabetes (0=No, 1=Yes)',
            'ADepress': 'Depression (0=No, 1=Yes)',
            'AAnxiety': 'Anxiety (0=No, 1=Yes)',
            'AAlcRate': 'Alcohol use rate',
            'AAlcNbDr': 'Number of drinks per occasion',
            'AAlc6Mor': 'Six or more drinks (0=No, 1=Yes)',
            'AI2RhADa': 'Days from injury to rehab admission',
            'ATrmEtio': 'Traumatic etiology code',
            'AAsscInj': 'Associated injuries',
            'AVertInj': 'Vertebral injury',
            'ASpinSrg': 'Spinal surgery (0=No, 1=Yes)',
            'AUMVAdm': 'Use of mechanical ventilation at admission',
            'AFScorRb': 'Functional independence score at rehab',
            'AASATotA': 'ASIA Motor Score at Admission (0-100)',
            'AASAImAd': 'ASIA Impairment Grade at Admission (A/B/C/D)',
            'ANurLvlA': 'Neurological level at admission'
        }
        return feature_descriptions
    
    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict discharge motor score from admission data
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary with feature names as keys and values as patient data
            
        Returns:
        --------
        dict with prediction results
        """
        # Create DataFrame with patient data
        df = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Encode categorical variables (same as training)
        from sklearn.preprocessing import LabelEncoder
        categorical_columns = ['AInjAge', 'AASAImAd', 'ANurLvlA']
        for col in categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Apply imputation
        X = self.imputer.transform(df)
        
        # Make prediction
        predicted_score = self.model.predict(X)[0]
        
        # Get feature importances for this prediction (if needed)
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


def example_usage():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = MotorScorePredictor()
    
    # Example patient data (you would collect this from user input)
    example_patient = {
        'AInjAge': 45,              # 45 years old
        'ASex': 1,                  # Male
        'ARace': 1,                 # Race code
        'AHispnic': 0,              # Not Hispanic
        'AMarStIj': 2,              # Married
        'AEducLvl': 3,              # Some college
        'APrLvlSt': 1,              # English
        'AFmIncLv': 3,              # Middle income
        'APrimPay': 1,              # Private insurance
        'APResInj': 1,              # Home
        'ADiabete': 0,              # No diabetes
        'ADepress': 0,              # No depression
        'AAnxiety': 0,              # No anxiety
        'AAlcRate': 0,              # No alcohol
        'AAlcNbDr': 0,              # 0 drinks
        'AAlc6Mor': 0,              # No binge drinking
        'AI2RhADa': 25,             # 25 days to rehab
        'ATrmEtio': 1,              # Traumatic
        'AAsscInj': 0,              # No associated injuries
        'AVertInj': 1,              # Vertebral injury
        'ASpinSrg': 1,              # Had spinal surgery
        'AUMVAdm': 0,               # No ventilation
        'AFScorRb': 50,             # Functional score
        'AASATotA': 42,             # Motor score at admission: 42/100
        'AASAImAd': 'C',            # Grade C at admission
        'ANurLvlA': 'T4'            # Neurological level
    }
    
    # Get prediction with interpretation
    print(predictor.predict_with_interpretation(example_patient))
    
    # Or get raw prediction dict
    result = predictor.predict(example_patient)
    print("\nRaw prediction result:")
    print(result)


if __name__ == "__main__":
    print("="*76)
    print("ASIA MOTOR SCORE PREDICTOR - INFERENCE MODE")
    print("="*76)
    print("\nThis script loads the trained model and predicts discharge motor score")
    print("from admission patient data.\n")
    
    # Run example
    example_usage()
    
    print("\n" + "="*76)
    print("TO USE IN YOUR APPLICATION:")
    print("="*76)
    print("""
from predict_motor_score import MotorScorePredictor

# Initialize once
predictor = MotorScorePredictor()

# Make predictions
result = predictor.predict(patient_data_dict)
print(f"Predicted discharge score: {result['predicted_discharge_motor_score']}")

# Or get formatted interpretation
interpretation = predictor.predict_with_interpretation(patient_data_dict)
print(interpretation)
    """)

