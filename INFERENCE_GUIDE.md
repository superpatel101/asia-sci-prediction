# ASIA SCI Prediction Models - Inference Guide

## Overview

Two trained machine learning models are available for predicting outcomes in spinal cord injury patients:

1. **Motor Score Predictor** - Predicts discharge motor score (0-100) from admission data
2. **Impairment Grade Predictor** - Predicts discharge ASIA grade (A/B/C/D/E) from admission data

Both models use admission-time features only (no data leakage) and can be used for early patient counseling.

---

## Quick Start

### 1. Motor Score Prediction

```python
from predict_motor_score import MotorScorePredictor

# Initialize predictor
predictor = MotorScorePredictor()

# Prepare patient data (all 26 features required)
patient_data = {
    'AInjAge': 45,              # Age at injury (years)
    'ASex': 1,                  # Sex (1=Male, 2=Female)
    'ARace': 1,                 # Race code
    'AHispnic': 0,              # Hispanic (0=No, 1=Yes)
    'AMarStIj': 2,              # Marital status
    'AEducLvl': 3,              # Education level
    'APrLvlSt': 1,              # Primary language
    'AFmIncLv': 3,              # Family income level
    'APrimPay': 1,              # Primary payer
    'APResInj': 1,              # Place of residence
    'ADiabete': 0,              # Diabetes (0=No, 1=Yes)
    'ADepress': 0,              # Depression (0=No, 1=Yes)
    'AAnxiety': 0,              # Anxiety (0=No, 1=Yes)
    'AAlcRate': 0,              # Alcohol use rate
    'AAlcNbDr': 0,              # Number of drinks
    'AAlc6Mor': 0,              # Binge drinking (0=No, 1=Yes)
    'AI2RhADa': 25,             # Days from injury to rehab
    'ATrmEtio': 1,              # Traumatic etiology
    'AAsscInj': 0,              # Associated injuries
    'AVertInj': 1,              # Vertebral injury
    'ASpinSrg': 1,              # Spinal surgery (0=No, 1=Yes)
    'AUMVAdm': 0,               # Mechanical ventilation
    'AFScorRb': 50,             # Functional score at rehab
    'AASATotA': 42,             # Motor score at admission (0-100)
    'AASAImAd': 'C',            # ASIA grade at admission (A/B/C/D)
    'ANurLvlA': 'T4'            # Neurological level at admission
}

# Get prediction with formatted interpretation
print(predictor.predict_with_interpretation(patient_data))

# Or get raw prediction dictionary
result = predictor.predict(patient_data)
print(f"Predicted discharge score: {result['predicted_discharge_motor_score']}")
print(f"Expected improvement: {result['expected_improvement']:+.1f} points")
```

**Output:**
```
Predicted Discharge Motor Score: 52.8 / 100
Expected Improvement:            +10.8 points

INTERPRETATION:
  ✓ GOOD expected recovery (10-20 points)
```

---

### 2. Impairment Grade Prediction

```python
from predict_impairment_grade import ImpairmentGradePredictor

# Initialize predictor
predictor = ImpairmentGradePredictor()

# Use same patient_data dictionary as above

# Get prediction with formatted interpretation
print(predictor.predict_with_interpretation(patient_data))

# Or get raw prediction dictionary
result = predictor.predict(patient_data)
print(f"Predicted grade: {result['predicted_discharge_grade']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"All grade probabilities: {result['grade_probabilities']}")
```

**Output:**
```
Predicted Discharge Grade: D
Confidence: 65.3%

PROBABILITIES FOR ALL GRADES:
  Grade D: ██████████████████████████░░░░░░░░░░░░░░  65.3% ← PREDICTED
  Grade C: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  24.1%
  Grade E: ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.2%
  ...
```

---

## Required Features (26 total)

Both models require the same 26 admission-time features:

### Demographics & Social (10 features):
- `AInjAge` - Age at injury (years)
- `ASex` - Sex (1=Male, 2=Female, 3=Other, 9=Unknown)
- `ARace` - Race code
- `AHispnic` - Hispanic ethnicity (0=No, 1=Yes, 9=Unknown)
- `AMarStIj` - Marital status at injury
- `AEducLvl` - Education level
- `APrLvlSt` - Primary language spoken
- `AFmIncLv` - Family income level
- `APrimPay` - Primary payer (insurance)
- `APResInj` - Place of residence before injury

### Medical History (6 features):
- `ADiabete` - Diabetes (0=No, 1=Yes)
- `ADepress` - Depression (0=No, 1=Yes)
- `AAnxiety` - Anxiety (0=No, 1=Yes)
- `AAlcRate` - Alcohol use rate
- `AAlcNbDr` - Number of drinks per occasion
- `AAlc6Mor` - Six or more drinks per occasion (0=No, 1=Yes)

### Injury & Treatment (7 features):
- `AI2RhADa` - Days from injury to rehabilitation admission
- `ATrmEtio` - Traumatic etiology code
- `AAsscInj` - Associated injuries
- `AVertInj` - Vertebral injury
- `ASpinSrg` - Spinal surgery (0=No, 1=Yes)
- `AUMVAdm` - Use of mechanical ventilation at admission
- `AFScorRb` - Functional independence score at rehab

### ASIA Scores (3 features):
- `AASATotA` - **ASIA Motor Score at Admission (0-100)** ⭐ Most important
- `AASAImAd` - **ASIA Impairment Grade at Admission (A/B/C/D)** ⭐ Most important
- `ANurLvlA` - Neurological level at admission (e.g., 'C5', 'T4')

---

## Model Performance

### Motor Score Model:
- **R² Score:** 0.8122 (explains 81.2% of variance)
- **RMSE:** 11.7 points
- **MAE:** 7.6 points
- **Training data:** 10,543 patients
- **No data leakage** - uses admission features only

**Clinical Utility:**
- Grade C patients: ~26 points average improvement
- Grade D patients: ~12 points average improvement
- Grade B patients: ~15 points average improvement
- Grade A patients: ~4 points average improvement

### Impairment Grade Model:
- **Accuracy:** 0.7537 (75.4%)
- **F1-Score (Weighted):** 0.7433
- **AUC (Weighted):** 0.9178
- **Training data:** 15,053 patients

**Per-Grade Performance:**
- Grade A: F1=0.73, Precision=0.76, Recall=0.71
- Grade B: F1=0.56, Precision=0.59, Recall=0.53
- Grade C: F1=0.76, Precision=0.77, Recall=0.75
- Grade D: F1=0.68, Precision=0.71, Recall=0.66
- Grade E: F1=0.84, Precision=0.81, Recall=0.88

---

## Important Notes

### Data Encoding:
- Categorical variables are automatically encoded by the scripts
- For `AASAImAd`: use 'A', 'B', 'C', or 'D' (letters)
- For `ANurLvlA`: use standard notation (e.g., 'C5', 'T4', 'L1')
- All other variables should be numeric

### Predictions:
- **Confidence intervals not provided** - use with clinical judgment
- Models trained on NSCISC database (North American population)
- Best used for **early counseling** about expected outcomes
- Not a replacement for clinical assessment

### Known Limitations:
1. **Motor Score Model:**
   - R² of 0.38 means 62% of variance unexplained
   - Individual outcomes can vary significantly
   - Works best for typical cases, less reliable for outliers

2. **Impairment Grade Model:**
   - Grade B has lower accuracy (F1=0.56) - harder to distinguish
   - Confidence < 60% suggests uncertain prediction
   - Works best when admission grade is A, C, or E

---

## Web Integration

To integrate into a web application:

### Flask Example:

```python
from flask import Flask, request, jsonify
from predict_motor_score import MotorScorePredictor
from predict_impairment_grade import ImpairmentGradePredictor

app = Flask(__name__)

# Initialize predictors once at startup
motor_predictor = MotorScorePredictor()
grade_predictor = ImpairmentGradePredictor()

@app.route('/predict/motor', methods=['POST'])
def predict_motor():
    patient_data = request.json
    try:
        result = motor_predictor.predict(patient_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/grade', methods=['POST'])
def predict_grade():
    patient_data = request.json
    try:
        result = grade_predictor.predict(patient_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

### HTML Form Example:

See `web_interface_example.html` for a complete HTML form that collects all 26 features and displays predictions.

---

## Testing

Run the example scripts to test:

```bash
# Test motor score predictor
python3 predict_motor_score.py

# Test impairment grade predictor  
python3 predict_impairment_grade.py
```

Both scripts include example patients and will display formatted predictions.

---

## Files Required

Make sure these files are in the same directory:

**Motor Score Model:**
- `predict_motor_score.py` - Inference script
- `random_forest_motor_clean_model.pkl` - Trained model (45 MB)
- `motor_clean_imputer.pkl` - Data imputer (1 KB)
- `motor_clean_feature_names.pkl` - Feature names (1 KB)

**Impairment Grade Model:**
- `predict_impairment_grade.py` - Inference script
- `random_forest_impairment_classifier.pkl` - Trained model (55 MB)
- `impairment_imputer.pkl` - Data imputer (1 KB)
- `impairment_feature_names.pkl` - Feature names (1 KB)

---

## Support

For questions or issues:
- Check the data dictionary: `NSCISC_Data_Dictionary_Viewer.html`
- Review training scripts: `train_motor_clean_model.py` and `train_impairment_classifier.py`
- See analysis reports in the working directory

---

## Citation

If you use these models in research or clinical applications, please cite:

```
ASIA Spinal Cord Injury Outcome Prediction Models
Trained on the National Spinal Cord Injury Statistical Center (NSCISC) Database
Models: Random Forest Regressor (Motor Score) and Random Forest Classifier (Impairment Grade)
Date: October 2024
```

---

## License & Disclaimer

These models are for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals using comprehensive patient assessment. The models provide statistical predictions based on historical data and should not be the sole basis for treatment decisions.

