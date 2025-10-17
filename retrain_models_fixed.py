"""
Retrain both models with FIXED categorical encoding that can be reproduced during inference
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING MODELS WITH FIXED CATEGORICAL ENCODING")
print("="*80)

# ============================================================================
# MOTOR SCORE MODEL
# ============================================================================
print("\n" + "="*80)
print("1. TRAINING MOTOR SCORE PREDICTION MODEL")
print("="*80)

df_motor = pd.read_csv('V2_EDIT_modelreadyASIAMotor.csv')
print(f"Motor dataset shape: {df_motor.shape}")

# Separate features and target
target_column = 'AASATotD'
all_columns = df_motor.columns.tolist()
admission_features = []
discharge_features = []

for col in all_columns:
    if col == target_column:
        continue
    col_lower = col.lower()
    if 'dis' in col_lower or col.endswith('Ds') or col.endswith('D'):
        if col != 'ADiabete':
            discharge_features.append(col)
    else:
        admission_features.append(col)

print(f"✓ Using {len(admission_features)} admission features")
print(f"✓ Excluding {len(discharge_features)} discharge features")

X = df_motor[admission_features].copy()
y = df_motor[target_column]

# Remove rows with missing target
mask = ~y.isna()
X = X[mask]
y = y[mask]

print(f"✓ Dataset: {X.shape[0]} patients")

# EXPLICIT CATEGORICAL ENCODING (consistent for training and inference)
def encode_categorical_features(df):
    """Apply explicit categorical encoding"""
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
    
    df['ANurLvlA'] = df['ANurLvlA'].apply(encode_neuro_level)
    
    return df

print("\nApplying explicit categorical encoding...")
X = encode_categorical_features(X)
print("✓ Categorical encoding applied")

# Impute missing values
imputer_motor = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer_motor.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Train model
print("\nTraining Random Forest Regressor...")
rf_motor = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_motor.fit(X_train, y_train)

# Evaluate
y_test_pred = rf_motor.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n✓ MOTOR SCORE MODEL PERFORMANCE:")
print(f"  R² = {test_r2:.4f}")
print(f"  RMSE = {test_rmse:.2f}")
print(f"  MAE = {test_mae:.2f}")

# Save artifacts
joblib.dump(rf_motor, 'random_forest_motor_clean_model.pkl')
joblib.dump(imputer_motor, 'motor_clean_imputer.pkl')
joblib.dump(X_train.columns.tolist(), 'motor_clean_feature_names.pkl')
print("✓ Motor model saved")

# ============================================================================
# IMPAIRMENT GRADE MODEL
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING IMPAIRMENT GRADE PREDICTION MODEL")
print("="*80)

df_imp = pd.read_csv('ModelreadyAISMedsurgtodischarge.csv')
print(f"Impairment dataset shape: {df_imp.shape}")

# Separate features and target
target_column_imp = 'AASAImDs'
X_imp = df_imp.drop(columns=[target_column_imp])
y_imp = df_imp[target_column_imp]

# Remove rows with missing target
mask = ~y_imp.isna()
X_imp = X_imp[mask]
y_imp = y_imp[mask].astype(int)

print(f"✓ Dataset: {X_imp.shape[0]} patients")
print(f"✓ Classes: {sorted(y_imp.unique())}")

# Apply same categorical encoding
print("\nApplying explicit categorical encoding...")
X_imp = encode_categorical_features(X_imp)
print("✓ Categorical encoding applied")

# Impute missing values
imputer_imp = SimpleImputer(strategy='median')
X_imp_imputed = pd.DataFrame(imputer_imp.fit_transform(X_imp), columns=X_imp.columns)

# Split data (stratified)
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(
    X_imp_imputed, y_imp, test_size=0.2, random_state=42, stratify=y_imp
)

print(f"✓ Train: {X_train_imp.shape[0]}, Test: {X_test_imp.shape[0]}")

# Train model
print("\nTraining Random Forest Classifier...")
rf_imp = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_imp.fit(X_train_imp, y_train_imp)

# Evaluate
y_test_pred_imp = rf_imp.predict(X_test_imp)
test_acc = accuracy_score(y_test_imp, y_test_pred_imp)
test_f1 = f1_score(y_test_imp, y_test_pred_imp, average='weighted')

print(f"\n✓ IMPAIRMENT GRADE MODEL PERFORMANCE:")
print(f"  Accuracy = {test_acc:.4f}")
print(f"  F1-Score = {test_f1:.4f}")

# Save artifacts
joblib.dump(rf_imp, 'random_forest_impairment_classifier.pkl')
joblib.dump(imputer_imp, 'impairment_imputer.pkl')
joblib.dump(X_train_imp.columns.tolist(), 'impairment_feature_names.pkl')
print("✓ Impairment model saved")

print("\n" + "="*80)
print("✓ BOTH MODELS RETRAINED AND SAVED WITH FIXED ENCODING")
print("="*80)
print("\nNow run the Flask app to test predictions!")

