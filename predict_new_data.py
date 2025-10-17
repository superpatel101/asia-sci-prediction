"""
Script to make predictions using the trained Random Forest model.
This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib

def load_model_artifacts():
    """Load the trained model and preprocessing artifacts."""
    print("Loading model artifacts...")
    model = joblib.load('random_forest_asia_motor_model.pkl')
    imputer = joblib.load('imputer.pkl')
    feature_names = joblib.load('feature_names.pkl')
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
    categorical_columns = ['AInjAge', 'AASAImAd', 'AASAImDs', 'ANurLvlA', 'ANurLvlD']
    for col in categorical_columns:
        if col in data.columns:
            if data[col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
    
    # Apply imputation
    data_imputed = pd.DataFrame(imputer.transform(data), columns=feature_names)
    
    return data_imputed

def predict(data_path=None, data_df=None):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to CSV file containing new data
    data_df : pd.DataFrame, optional
        DataFrame containing new data
        
    Returns:
    --------
    predictions : array
        Predicted AASATotD values
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
    predictions = model.predict(data_processed)
    
    print("✓ Predictions completed!")
    return predictions

def predict_single_sample(sample_dict):
    """
    Make prediction for a single sample.
    
    Parameters:
    -----------
    sample_dict : dict
        Dictionary containing feature values for a single sample
        
    Returns:
    --------
    prediction : float
        Predicted AASATotD value
    """
    df = pd.DataFrame([sample_dict])
    prediction = predict(data_df=df)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("RANDOM FOREST MODEL - PREDICTION SCRIPT")
    print("="*60)
    
    # Example 1: Predict on the entire test set
    print("\n[Example 1] Making predictions on the original dataset...")
    df = pd.read_csv('V2_EDIT_modelreadyASIAMotor.csv')
    
    # Remove target column if it exists
    if 'AASATotD' in df.columns:
        X_test = df.drop(columns=['AASATotD'])
        y_true = df['AASATotD']
    else:
        X_test = df
        y_true = None
    
    # Make predictions
    predictions = predict(data_df=X_test)
    
    # Display results
    print(f"\nFirst 10 predictions:")
    results_df = pd.DataFrame({
        'Predicted_AASATotD': predictions[:10]
    })
    
    if y_true is not None:
        results_df['Actual_AASATotD'] = y_true[:10].values
        results_df['Error'] = results_df['Actual_AASATotD'] - results_df['Predicted_AASATotD']
    
    print(results_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION STATISTICS")
    print(f"{'='*60}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean prediction: {predictions.mean():.2f}")
    print(f"Std prediction: {predictions.std():.2f}")
    print(f"Min prediction: {predictions.min():.2f}")
    print(f"Max prediction: {predictions.max():.2f}")
    
    if y_true is not None:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE ON FULL DATASET")
        print(f"{'='*60}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    # Save predictions
    output_df = X_test.copy()
    output_df['Predicted_AASATotD'] = predictions
    if y_true is not None:
        output_df['Actual_AASATotD'] = y_true.values
        output_df['Prediction_Error'] = y_true.values - predictions
    
    output_file = 'predictions_output.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Full predictions saved to '{output_file}'")
    
    print(f"\n{'='*60}")
    print("✓ PREDICTION COMPLETE!")
    print(f"{'='*60}\n")
    
    # Example 2: How to predict for a single new patient
    print("\n" + "="*60)
    print("[Example 2] Predicting for a single patient")
    print("="*60)
    print("\nTo predict for a new patient, create a dictionary with all features:")
    print("""
# Example:
new_patient = {
    'AInjAge': 28,
    'ASex': 1,
    'ARace': 1,
    'AHispnic': 0,
    'AMarStIj': 3,
    'AEducLvl': 2,
    'APrLvlSt': 1,
    'AFmIncLv': 9,
    'APrimPay': 4,
    'APResInj': 99,
    'APResDis': 6,
    'ADiabete': 9,
    'ADepress': 9,
    'AAnxiety': 9,
    'AAlcRate': 9,
    'AAlcNbDr': 9,
    'AAlc6Mor': 9,
    'AI2RhADa': 9,
    'ATrmEtio': 30,
    'AAsscInj': 9,
    'AVertInj': 9,
    'ASpinSrg': 1,
    'AUMVAdm': 0,
    'AUMVDis': 0,
    'ABdMMDis': 5,
    'AFScorRb': 99,
    'AFScorDs': 99,
    'AASATotA': 11.0,
    'AASAImAd': 1,
    'AASAImDs': 1,
    'ANurLvlA': 'C05',
    'ANurLvlD': 'C06'
}

prediction = predict_single_sample(new_patient)
print(f"Predicted AASATotD: {prediction:.2f}")
""")

