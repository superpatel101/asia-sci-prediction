# ASIA Impairment Grade Prediction - Random Forest Classifier

This project implements a **Random Forest Classification model** to predict ASIA Impairment Grade at discharge (`AASAImDs`) for patients with traumatic spinal cord injuries.

---

## 📊 Model Performance

The trained classifier achieves excellent performance on the spinal injury dataset:

| Metric | Training Set | Test Set | Cross-Validation (5-fold) |
|--------|--------------|----------|---------------------------|
| **Accuracy** | 93.51% | **82.60%** | 83.44% ± 0.69% |
| **F1-Score (Macro)** | 89.73% | **65.89%** | - |
| **F1-Score (Weighted)** | 93.56% | **82.34%** | - |
| **Precision (Weighted)** | - | **82.68%** | - |
| **Recall (Weighted)** | - | **82.60%** | - |
| **AUC (Weighted)** | - | **94.17%** | - |

### What does this mean?
- The model correctly classifies **82.6%** of patients into the correct ASIA grade
- Weighted F1-score of **82.3%** balances precision and recall
- AUC of **94.2%** indicates excellent discrimination ability
- Performance is consistent across cross-validation folds

---

## 🎯 ASIA Impairment Scale

The model predicts one of 5 ASIA impairment grades:

| Grade | Numeric | Description | Dataset % |
|-------|---------|-------------|-----------|
| **A** | 1 | Complete - No motor or sensory function preserved | 10.35% |
| **B** | 2 | Incomplete - Sensory but no motor function preserved | 9.98% |
| **C** | 3 | Incomplete - Motor function preserved, <50% key muscles against gravity | 30.27% |
| **D** | 4 | Incomplete - Motor function preserved, ≥50% key muscles against gravity | 0.70% |
| **E** | 5 | Normal - Motor and sensory function normal | 48.70% |

**Note:** Class D is severely underrepresented (only 0.70% of data), which affects its prediction accuracy.

---

## 📈 Per-Class Performance (Test Set)

| Grade | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **A** | 0.61 | 0.72 | 0.66 | 312 |
| **B** | 0.52 | 0.46 | 0.49 | 300 |
| **C** | 0.89 | 0.75 | 0.81 | 912 |
| **D** | 0.36 | 0.43 | 0.39 | 21 |
| **E** | 0.90 | 0.98 | 0.94 | 1466 |

**Key Insights:**
- **Best performance:** Grade E (normal function) - F1: 0.94
- **Good performance:** Grade C (incomplete, motor) - F1: 0.81
- **Moderate performance:** Grade A (complete) - F1: 0.66
- **Challenges:** Grades B and D have lower scores due to class imbalance

---

## 📁 Project Files

### Core Model Files
- `random_forest_impairment_classifier.pkl` - Trained Random Forest classifier
- `impairment_imputer.pkl` - Data preprocessing imputer
- `impairment_feature_names.pkl` - Feature names for predictions

### Scripts
- `train_impairment_classifier.py` - Complete training pipeline
- `predict_impairment.py` - Prediction script for new patients

### Outputs
- `impairment_model_summary.txt` - Detailed performance report
- `impairment_feature_importance.csv` - Feature importance rankings
- `impairment_predictions_output.csv` - Predictions for all dataset samples

### Visualizations
- `impairment_confusion_matrix.png` - Confusion matrix heatmap
- `impairment_feature_importance.png` - Top 20 feature importance chart
- `impairment_class_distributions.png` - Actual vs predicted distributions
- `impairment_per_class_performance.png` - Per-class metrics comparison

---

## 🔑 Top Predictive Features

The model identified these features as most important for predicting discharge impairment:

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | `AASAImAd` | 38.27% | **ASIA impairment at admission** |
| 2 | `AI2RhADa` | 14.18% | Injury to rehab admission days |
| 3 | `AUMVAdm` | 7.47% | Upper motor vehicle at admission |
| 4 | `ANurLvlA` | 7.44% | Neurological level at admission |
| 5 | `AInjAge` | 7.20% | Age at injury |
| 6 | `ATrmEtio` | 4.25% | Trauma etiology |
| 7 | `AASATotA` | 3.81% | ASIA total score at admission |
| 8 | `AEducLvl` | 3.41% | Education level |
| 9 | `APrimPay` | 2.27% | Primary payer |
| 10 | `AMarStIj` | 2.26% | Marital status at injury |

**Critical Insight:** The admission impairment grade (`AASAImAd`) is by far the strongest predictor (38.3%), indicating that initial injury severity is the primary determinant of discharge impairment.

---

## 🚀 Quick Start Guide

### 1. Training the Model

```bash
python3 train_impairment_classifier.py
```

This will:
- Load and preprocess the dataset (15,053 patients)
- Train a Random Forest classifier with 200 trees
- Use stratified sampling to handle class imbalance
- Evaluate performance with train/test split and cross-validation
- Generate visualizations and reports
- Save the trained model and artifacts

### 2. Making Predictions

#### Option A: Predict on a CSV file

```python
from predict_impairment import predict, ASIA_GRADE_MAP

# Predict on new data from CSV
predictions = predict(data_path='path/to/new_data.csv')

# Convert numeric to letter grades
letter_grades = [ASIA_GRADE_MAP[int(p)] for p in predictions]
print(letter_grades)
```

#### Option B: Predict with probabilities

```python
from predict_impairment import predict, ASIA_GRADE_MAP

# Get class probabilities
probabilities = predict(data_path='new_patients.csv', return_proba=True)

# Get predicted grade and confidence
for i, proba in enumerate(probabilities):
    predicted_grade = proba.argmax() + 1  # +1 because grades are 1-5
    confidence = proba.max()
    grade_letter = ASIA_GRADE_MAP[predicted_grade]
    print(f"Patient {i}: Grade {grade_letter} (confidence: {confidence:.2%})")
```

#### Option C: Predict for a single patient

```python
from predict_impairment import predict_single_sample, ASIA_GRADE_MAP, interpret_prediction

# Define patient features
new_patient = {
    'AInjAge': 26,
    'ASex': 1,
    'ARace': 1,
    'AHispnic': 0,
    'AMarStIj': 1,
    'AEducLvl': 2,
    'APrLvlSt': 7,
    'AFmIncLv': 9,
    'APrimPay': 3,
    'APResInj': 99,
    'ADiabete': 9,
    'ADepress': 9,
    'AAnxiety': 9,
    'AAlcRate': 9,
    'AAlcNbDr': 9,
    'AAlc6Mor': 9,
    'AI2RhADa': 51,
    'ATrmEtio': 20,
    'AAsscInj': 9,
    'AVertInj': 9,
    'ASpinSrg': 9,
    'AUMVAdm': 2,
    'AFScorRb': 99,
    'AASATotA': 999.0,
    'AASAImAd': 5,      # Admission impairment (most important!)
    'ANurLvlA': 'C04'   # Neurological level
}

# Get prediction
prediction = predict_single_sample(new_patient)
probabilities = predict_single_sample(new_patient, return_proba=True)

# Interpret
grade_letter, interpretation = interpret_prediction(prediction)
print(f"Predicted Grade: {grade_letter}")
print(f"Interpretation: {interpretation}")
print(f"Confidence: {probabilities.max():.2%}")
```

### 3. Loading the Model Directly

```python
import joblib

# Load the trained model
model = joblib.load('random_forest_impairment_classifier.pkl')

# Load preprocessing artifacts
imputer = joblib.load('impairment_imputer.pkl')
feature_names = joblib.load('impairment_feature_names.pkl')

# Make predictions
# predictions = model.predict(preprocessed_data)
# probabilities = model.predict_proba(preprocessed_data)
```

---

## 📋 Data Requirements

The model expects **26 input features**:

### Demographics (10 features)
- `AInjAge`, `ASex`, `ARace`, `AHispnic`, `AMarStIj`, `AEducLvl`, `APrLvlSt`, `AFmIncLv`, `APrimPay`, `APResInj`

### Medical History (6 features)
- `ADiabete`, `ADepress`, `AAnxiety`, `AAlcRate`, `AAlcNbDr`, `AAlc6Mor`

### Injury & Clinical Details (10 features)
- `AI2RhADa`, `ATrmEtio`, `AAsscInj`, `AVertInj`, `ASpinSrg`, `AUMVAdm`, `AFScorRb`, `AASATotA`, `AASAImAd`, `ANurLvlA`

**Note:** Missing values are automatically handled by the imputer using median imputation.

---

## 🔬 Model Architecture

### Algorithm: Random Forest Classifier

**Hyperparameters:**
- Number of trees: 200
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2
- Max features: 'sqrt'
- Class weight: 'balanced' (handles imbalance)
- Random state: 42

### Training Configuration
- Train/Test split: 80/20 (stratified)
- Cross-validation: 5-fold stratified
- Feature preprocessing: Median imputation for missing values
- Categorical encoding: Label encoding for categorical features

---

## 📊 Dataset Statistics

- **Total samples:** 15,053 patients
- **Training set:** 12,042 samples (80%)
- **Test set:** 3,011 samples (20%)
- **Features:** 26 input features
- **Target variable:** `AASAImDs` (ASIA impairment grade at discharge)
  - Classes: 1-5 (Grades A-E)
  - Most common: Grade E (48.70%)
  - Least common: Grade D (0.70%)

---

## 🛠️ Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Insights

### What the Model Tells Us

1. **Admission impairment is most predictive** - The ASIA impairment grade at admission (`AASAImAd`) accounts for 38.3% of the model's predictive power. Initial severity is the strongest indicator of discharge status.

2. **Time to rehabilitation matters** - Days from injury to rehab admission (`AI2RhADa`) is the second most important feature (14.2%), suggesting early intervention may impact outcomes.

3. **Neurological level is critical** - Both the neurological level at admission (`ANurLvlA`) and age at injury (`AInjAge`) contribute significantly (7.4% and 7.2%).

4. **Grade E is easiest to predict** - Normal function patients are classified with 98% recall and 90% precision.

5. **Grade D is challenging** - With only 105 samples (0.70% of data), Grade D predictions have lower accuracy (F1: 0.39).

### Clinical Applications

This model can be used to:
- **Predict discharge impairment early** - Use admission data to forecast outcomes
- **Identify high-risk patients** - Those predicted for severe impairment may need intensive care
- **Plan resource allocation** - Anticipate rehabilitation needs based on predicted grades
- **Set realistic expectations** - Help patients and families understand likely outcomes
- **Benchmark performance** - Compare actual vs predicted outcomes to assess care quality

---

## 🎯 Confusion Matrix Insights

The confusion matrix (see `impairment_confusion_matrix.png`) reveals:

**Grade E (Normal):**
- Excellent prediction: 1,431/1,466 correct (98%)
- Rarely misclassified

**Grade C (Incomplete Motor):**
- Good prediction: 684/912 correct (75%)
- Sometimes confused with Grade B or A (more severe)

**Grade A (Complete):**
- Moderate prediction: 224/312 correct (72%)
- 71 patients misclassified as Grade E (overly optimistic)

**Grade B (Incomplete Sensory):**
- Challenging: 139/300 correct (46%)
- Often confused with Grades A and C

**Grade D (Incomplete Motor, ≥50%):**
- Most difficult: 9/21 correct (43%)
- Very low sample size makes learning difficult

---

## ⚠️ Model Limitations & Considerations

1. **Class imbalance:** Grade D has only 105 samples (0.70%), leading to poor predictions for this class
2. **Admission features:** Requires admission impairment grade, which may not always be immediately available
3. **Dataset-specific:** Trained on specific population; may need retraining for different demographics
4. **Point estimates:** Provides single predictions; use probabilities to assess confidence
5. **No temporal modeling:** Doesn't capture recovery trajectory over time

---

## 💡 Recommendations for Use

### High Confidence Predictions
Use model predictions directly when:
- ✅ Confidence > 80%
- ✅ Predicting Grades C or E
- ✅ Patient characteristics match training data

### Lower Confidence Predictions
Exercise caution when:
- ⚠️ Confidence < 60%
- ⚠️ Predicting Grades B or D
- ⚠️ Unusual patient characteristics
- ⚠️ Multiple grades have similar probabilities

### Best Practices
1. **Always check probabilities** - Don't rely on single prediction alone
2. **Consider clinical context** - Model is a decision support tool, not replacement for clinical judgment
3. **Monitor calibration** - Periodically validate predictions against actual outcomes
4. **Update regularly** - Retrain model as new data becomes available

---

## 📈 Example Output

```
Patient Analysis:
  Predicted Grade: C (3)
  Confidence: 88.04%
  Interpretation: Incomplete - Motor function preserved,
                  less than half key muscles can move against gravity
  
  Class Probabilities:
    Grade A: 4.2%
    Grade B: 5.8%
    Grade C: 88.0%  ← Predicted
    Grade D: 0.0%
    Grade E: 2.0%
```

---

## 🤝 Usage Example

Complete end-to-end example:

```python
import pandas as pd
from predict_impairment import predict, ASIA_GRADE_MAP

# Load new patient data
new_patients = pd.read_csv('new_patients.csv')

# Make predictions
predictions = predict(data_df=new_patients)
probabilities = predict(data_df=new_patients, return_proba=True)

# Add predictions to dataframe
new_patients['Predicted_Grade_Numeric'] = predictions.astype(int)
new_patients['Predicted_Grade_Letter'] = [ASIA_GRADE_MAP[int(p)] for p in predictions]
new_patients['Prediction_Confidence'] = [proba.max() for proba in probabilities]

# Flag low confidence predictions
new_patients['Low_Confidence'] = new_patients['Prediction_Confidence'] < 0.60

# Save results
new_patients.to_csv('predicted_impairments.csv', index=False)

print(f"Predicted {len(predictions)} patient outcomes")
print(f"\nGrade distribution:")
print(new_patients['Predicted_Grade_Letter'].value_counts().sort_index())
print(f"\nLow confidence predictions: {new_patients['Low_Confidence'].sum()}")
```

---

## 📧 Support

For questions or issues with the model, please refer to:
- `impairment_model_summary.txt` - Detailed performance metrics
- `impairment_feature_importance.csv` - Complete feature rankings
- Visualization files (.png) - Model behavior analysis

---

**Model trained on:** October 15, 2025  
**Algorithm:** Random Forest Classifier (Stratified, Balanced Classes)  
**Framework:** scikit-learn  
**Dataset:** 15,053 traumatic spinal cord injury patients  
**Classes:** ASIA Grades A, B, C, D, E (1-5)  
**Performance:** 82.6% accuracy, 94.2% AUC  
**License:** For research and clinical use

