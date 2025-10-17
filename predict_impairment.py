"""
Script to make predictions using the trained Random Forest Impairment Classifier.
Predicts ASIA Impairment Grade at Discharge (AASAImDs).
"""

import pandas as pd
import numpy as np
import joblib

# ASIA Grade mapping
ASIA_GRADE_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

def load_model_artifacts():
    """Load the trained model and preprocessing artifacts."""
    print("Loading model artifacts...")
    model = joblib.load('random_forest_impairment_classifier.pkl')
    imputer = joblib.load('impairment_imputer.pkl')
    feature_names = joblib.load('impairment_feature_names.pkl')
    print("✓ Model artifacts loaded successfully!")
    return model, imputer, feature_names

def preprocess_data(data, imputer, feature_names):
    """Preprocess new data using the saved imputer."""
    # Ensure data has the correct features
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features in the correct order
    data = data[feature_names]
    
    # Handle categorical columns (label encoding)
    categorical_columns = ['AInjAge', 'AASAImAd', 'ANurLvlA']
    for col in categorical_columns:
        if col in data.columns:
            if data[col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
    
    # Apply imputation
    data_imputed = pd.DataFrame(imputer.transform(data), columns=feature_names)
    
    return data_imputed

def predict(data_path=None, data_df=None, return_proba=False):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to CSV file containing new data
    data_df : pd.DataFrame, optional
        DataFrame containing new data
    return_proba : bool, default=False
        If True, return class probabilities instead of class labels
        
    Returns:
    --------
    predictions : array
        Predicted ASIA Impairment grades (1-5) or probabilities
    """
    # Load model artifacts
    model, imputer, feature_names = load_model_artifacts()
    
    # Load data
    if data_path is not None:
        print(f"\nLoading data from: {data_path}")
        data = pd.read_csv(data_path)
    elif data_df is not None:
        print("\nUsing provided DataFrame")
        data = data_df.copy()
    else:
        raise ValueError("Either data_path or data_df must be provided")
    
    print(f"Data shape: {data.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    data_processed = preprocess_data(data, imputer, feature_names)
    
    # Make predictions
    print("Making predictions...")
    if return_proba:
        predictions = model.predict_proba(data_processed)
    else:
        predictions = model.predict(data_processed)
    
    print("✓ Predictions completed!")
    return predictions

def predict_single_sample(sample_dict, return_proba=False):
    """
    Make prediction for a single sample.
    
    Parameters:
    -----------
    sample_dict : dict
        Dictionary containing feature values for a single sample
    return_proba : bool, default=False
        If True, return class probabilities
        
    Returns:
    --------
    prediction : int or array
        Predicted ASIA Impairment grade or probabilities
    """
    df = pd.DataFrame([sample_dict])
    predictions = predict(data_df=df, return_proba=return_proba)
    return predictions[0]

def interpret_prediction(grade_numeric):
    """Convert numeric grade to letter grade with interpretation."""
    grade_letter = ASIA_GRADE_MAP.get(int(grade_numeric), '?')
    
    interpretations = {
        'A': 'Complete - No motor or sensory function preserved',
        'B': 'Incomplete - Sensory but no motor function preserved',
        'C': 'Incomplete - Motor function preserved, less than half key muscles can move against gravity',
        'D': 'Incomplete - Motor function preserved, at least half key muscles can move against gravity',
        'E': 'Normal - Motor and sensory function normal'
    }
    
    return grade_letter, interpretations.get(grade_letter, 'Unknown')

# Example usage
if __name__ == "__main__":
    print("="*70)
    print("RANDOM FOREST CLASSIFIER - ASIA IMPAIRMENT PREDICTION")
    print("="*70)
    
    # Example: Predict on the entire dataset
    print("\n[Example 1] Making predictions on the original dataset...")
    df = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')
    
    # Remove target column if it exists
    if 'AASAImDs' in df.columns:
        X_test = df.drop(columns=['AASAImDs'])
        y_true = df['AASAImDs']
    else:
        X_test = df
        y_true = None
    
    # Make predictions
    predictions = predict(data_df=X_test)
    probabilities = predict(data_df=X_test, return_proba=True)
    
    # Display results
    print(f"\nFirst 10 predictions:")
    results_df = pd.DataFrame({
        'Predicted_Grade_Num': predictions[:10].astype(int),
        'Predicted_Grade_Letter': [ASIA_GRADE_MAP.get(int(p), '?') for p in predictions[:10]],
        'Confidence': [probabilities[i].max() for i in range(10)]
    })
    
    if y_true is not None:
        results_df['Actual_Grade_Num'] = y_true[:10].values.astype(int)
        results_df['Actual_Grade_Letter'] = [ASIA_GRADE_MAP.get(int(a), '?') for a in y_true[:10]]
        results_df['Correct'] = (results_df['Predicted_Grade_Num'] == results_df['Actual_Grade_Num'])
    
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("PREDICTION STATISTICS")
    print(f"{'='*70}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"\nPredicted grade distribution:")
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for grade_num, count in pred_counts.items():
        grade_letter = ASIA_GRADE_MAP.get(int(grade_num), '?')
        pct = (count / len(predictions)) * 100
        print(f"  Grade {grade_letter} ({int(grade_num)}): {count:5d} ({pct:5.2f}%)")
    
    if y_true is not None:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        accuracy = accuracy_score(y_true, predictions)
        f1_macro = f1_score(y_true, predictions, average='macro')
        f1_weighted = f1_score(y_true, predictions, average='weighted')
        
        print(f"\n{'='*70}")
        print("MODEL PERFORMANCE ON FULL DATASET")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, predictions, 
                                   target_names=[f"Grade {ASIA_GRADE_MAP.get(int(c), c)}" 
                                               for c in sorted(y_true.unique())]))
    
    # Save predictions
    output_df = X_test.copy()
    output_df['Predicted_AASAImDs_Numeric'] = predictions.astype(int)
    output_df['Predicted_AASAImDs_Letter'] = [ASIA_GRADE_MAP.get(int(p), '?') for p in predictions]
    output_df['Prediction_Confidence'] = [probabilities[i].max() for i in range(len(predictions))]
    
    # Add probability columns for each grade
    for i, grade in enumerate(sorted(ASIA_GRADE_MAP.keys())):
        output_df[f'Prob_Grade_{ASIA_GRADE_MAP[grade]}'] = probabilities[:, i]
    
    if y_true is not None:
        output_df['Actual_AASAImDs'] = y_true.values.astype(int)
        output_df['Correct_Prediction'] = (predictions.astype(int) == y_true.values.astype(int))
    
    output_file = 'impairment_predictions_output.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Full predictions saved to '{output_file}'")
    
    print(f"\n{'='*70}")
    print("✓ PREDICTION COMPLETE!")
    print(f"{'='*70}\n")
    
    # Example 2: Interpreting predictions
    print("\n" + "="*70)
    print("[Example 2] Prediction Interpretation")
    print("="*70)
    print("\nFor the first patient:")
    grade_num = int(predictions[0])
    grade_letter, interpretation = interpret_prediction(grade_num)
    confidence = probabilities[0].max()
    
    print(f"  Predicted Grade: {grade_letter} ({grade_num})")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Interpretation: {interpretation}")
    
    print(f"\n  Class Probabilities:")
    for i, (grade_num, grade_letter) in enumerate(ASIA_GRADE_MAP.items()):
        prob = probabilities[0][i]
        print(f"    Grade {grade_letter}: {prob:.2%}")
    
    # Example 3: How to predict for a single new patient
    print("\n" + "="*70)
    print("[Example 3] Predicting for a single patient")
    print("="*70)
    print("\nTo predict for a new patient, create a dictionary with all features:")
    print("""
# Example:
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
    'AASAImAd': 5,  # Admission impairment
    'ANurLvlA': 'C04'
}

# Get prediction
prediction = predict_single_sample(new_patient)
probabilities = predict_single_sample(new_patient, return_proba=True)

grade_letter = ASIA_GRADE_MAP[int(prediction)]
print(f"Predicted ASIA Impairment at Discharge: Grade {grade_letter} ({int(prediction)})")
print(f"Confidence: {probabilities.max():.2%}")
""")

