"""
ASIA Impairment Grade Prediction - Inference Script
Load trained model and predict discharge impairment grade from admission data
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ImpairmentGradePredictor:
    def __init__(self):
        """Load the trained impairment classifier and preprocessing artifacts"""
        print("Loading ASIA Impairment Grade Prediction Model...")
        self.model = joblib.load('random_forest_impairment_classifier.pkl')
        self.imputer = joblib.load('impairment_imputer.pkl')
        self.feature_names = joblib.load('impairment_feature_names.pkl')
        self.grade_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        self.grade_descriptions = {
            'A': 'Complete - No motor or sensory function preserved in S4-S5',
            'B': 'Sensory Incomplete - Sensory but not motor function preserved below neurological level',
            'C': 'Motor Incomplete - Motor function preserved below neurological level, <50% muscles have grade ≥3',
            'D': 'Motor Incomplete - Motor function preserved below neurological level, ≥50% muscles have grade ≥3',
            'E': 'Normal - Motor and sensory function is normal'
        }
        print(f"✓ Model loaded successfully")
        print(f"✓ Requires {len(self.feature_names)} features")
        print(f"✓ Predicts grades: {list(self.grade_map.values())}")
        
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
        Predict discharge impairment grade from admission data
        
        Parameters:
        -----------
        patient_data : dict
            Dictionary with feature names as keys and values as patient data
            
        Returns:
        --------
        dict with prediction results including probabilities for all grades
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
        predicted_class = self.model.predict(X)[0]
        predicted_grade = self.grade_map[predicted_class]
        
        # Get probabilities for all classes
        probabilities = self.model.predict_proba(X)[0]
        grade_probabilities = {
            self.grade_map[i+1]: float(prob) 
            for i, prob in enumerate(probabilities) if (i+1) in self.grade_map
        }
        
        # Get feature importances
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get admission grade for comparison
        admission_grade = patient_data.get('AASAImAd', 'Unknown')
        
        return {
            'predicted_discharge_grade': predicted_grade,
            'predicted_grade_description': self.grade_descriptions[predicted_grade],
            'admission_grade': admission_grade,
            'grade_probabilities': grade_probabilities,
            'confidence': float(probabilities.max()),
            'top_5_influential_features': top_features,
            'model_info': 'Random Forest Classifier trained on 15,053 patients'
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
  • Admission Motor Score:   {patient_data.get('AASATotA', 'Unknown')} / 100
  • Age:                     {patient_data.get('AInjAge', 'Unknown')} years
  • Days to Rehab:           {patient_data.get('AI2RhADa', 'Unknown')} days

PREDICTION:
  ┌─────────────────────────────────────────────────────────────┐
  │  Predicted Discharge Grade: {result['predicted_discharge_grade']}
  │  Confidence: {result['confidence']*100:.1f}%
  └─────────────────────────────────────────────────────────────┘

GRADE DESCRIPTION:
  {result['predicted_grade_description']}

PROBABILITIES FOR ALL GRADES:
"""
        
        # Sort probabilities by value
        sorted_probs = sorted(result['grade_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        
        for grade, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            marker = ' ← PREDICTED' if grade == result['predicted_discharge_grade'] else ''
            interpretation += f"  Grade {grade}: {bar} {prob*100:5.1f}%{marker}\n"
        
        # Interpretation based on grade change
        admission_grade = result['admission_grade']
        predicted_grade = result['predicted_discharge_grade']
        
        interpretation += "\nCLINICAL INTERPRETATION:\n"
        
        if admission_grade == predicted_grade:
            interpretation += f"  ○ Grade expected to remain {predicted_grade} (stable)\n"
        else:
            grade_order = ['A', 'B', 'C', 'D', 'E']
            if admission_grade in grade_order and predicted_grade in grade_order:
                adm_idx = grade_order.index(admission_grade)
                pred_idx = grade_order.index(predicted_grade)
                if pred_idx > adm_idx:
                    improvement = pred_idx - adm_idx
                    interpretation += f"  ✓ IMPROVEMENT expected: {admission_grade} → {predicted_grade} ({improvement} grade{'s' if improvement > 1 else ''})\n"
                else:
                    interpretation += f"  ⚠ Decline expected: {admission_grade} → {predicted_grade}\n"
        
        # Add context based on predicted grade
        if predicted_grade == 'A':
            interpretation += "  • Complete injury - Focus on adaptation and assistive technology\n"
        elif predicted_grade == 'B':
            interpretation += "  • Sensory incomplete - Some sensory preservation below injury level\n"
        elif predicted_grade == 'C':
            interpretation += "  • Motor incomplete <50% - Significant recovery potential with rehab\n"
        elif predicted_grade == 'D':
            interpretation += "  • Motor incomplete ≥50% - Good functional independence expected\n"
        elif predicted_grade == 'E':
            interpretation += "  • Normal function - Full motor and sensory recovery expected\n"
        
        interpretation += f"\nTOP 5 MOST INFLUENTIAL FEATURES FOR THIS PREDICTION:\n"
        for i, (feature, importance) in enumerate(result['top_5_influential_features'], 1):
            interpretation += f"  {i}. {feature}: {importance*100:.1f}% importance\n"
        
        interpretation += f"\n{result['model_info']}\n"
        interpretation += "="*76 + "\n"
        
        return interpretation


def example_usage():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = ImpairmentGradePredictor()
    
    # Example patient data (you would collect this from user input)
    example_patient = {
        'AInjAge': 35,              # 35 years old
        'ASex': 1,                  # Male
        'ARace': 1,                 # Race code
        'AHispnic': 0,              # Not Hispanic
        'AMarStIj': 1,              # Single
        'AEducLvl': 4,              # College graduate
        'APrLvlSt': 1,              # English
        'AFmIncLv': 4,              # Upper-middle income
        'APrimPay': 1,              # Private insurance
        'APResInj': 1,              # Home
        'ADiabete': 0,              # No diabetes
        'ADepress': 0,              # No depression
        'AAnxiety': 0,              # No anxiety
        'AAlcRate': 1,              # Occasional alcohol
        'AAlcNbDr': 2,              # 2 drinks
        'AAlc6Mor': 0,              # No binge drinking
        'AI2RhADa': 20,             # 20 days to rehab
        'ATrmEtio': 1,              # Traumatic
        'AAsscInj': 0,              # No associated injuries
        'AVertInj': 1,              # Vertebral injury
        'ASpinSrg': 1,              # Had spinal surgery
        'AUMVAdm': 0,               # No ventilation
        'AFScorRb': 65,             # Functional score
        'AASATotA': 45,             # Motor score at admission: 45/100
        'AASAImAd': 'C',            # Grade C at admission
        'ANurLvlA': 'T6'            # Neurological level
    }
    
    # Get prediction with interpretation
    print(predictor.predict_with_interpretation(example_patient))
    
    # Or get raw prediction dict
    result = predictor.predict(example_patient)
    print("\nRaw prediction result:")
    for key, value in result.items():
        if key != 'top_5_influential_features':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("="*76)
    print("ASIA IMPAIRMENT GRADE PREDICTOR - INFERENCE MODE")
    print("="*76)
    print("\nThis script loads the trained model and predicts discharge impairment")
    print("grade from admission patient data.\n")
    
    # Run example
    example_usage()
    
    print("\n" + "="*76)
    print("TO USE IN YOUR APPLICATION:")
    print("="*76)
    print("""
from predict_impairment_grade import ImpairmentGradePredictor

# Initialize once
predictor = ImpairmentGradePredictor()

# Make predictions
result = predictor.predict(patient_data_dict)
print(f"Predicted grade: {result['predicted_discharge_grade']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Probabilities: {result['grade_probabilities']}")

# Or get formatted interpretation
interpretation = predictor.predict_with_interpretation(patient_data_dict)
print(interpretation)
    """)

