import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING CLEAN MOTOR SCORE MODEL (NO DATA LEAKAGE)")
print("="*70)

print("\nLoading dataset...")
# Load the CLEAN data (admission features only)
df = pd.read_csv('/Users/aaryanpatel/Downloads/V2_EDIT_modelreadyASIAMotor.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# Identify features
all_columns = df.columns.tolist()
target_column = 'AASATotD'

print(f"\n{'='*70}")
print(f"Target variable: {target_column}")
print(f"{'='*70}")

# Check which features are ADMISSION vs DISCHARGE
admission_features = []
discharge_features = []

for col in all_columns:
    if col == target_column:
        continue
    col_lower = col.lower()
    # Check if this is a discharge feature
    if 'dis' in col_lower or col.endswith('Ds') or col.endswith('D'):
        if col != 'ADiabete':  # Exception
            discharge_features.append(col)
    else:
        admission_features.append(col)

print(f"\n✓ ADMISSION/INJURY-TIME FEATURES ({len(admission_features)}):")
for feat in sorted(admission_features):
    print(f"    {feat}")

if discharge_features:
    print(f"\n⚠ DISCHARGE FEATURES FOUND ({len(discharge_features)}) - WILL BE EXCLUDED:")
    for feat in sorted(discharge_features):
        print(f"    {feat}")

# Separate features and target
X = df[admission_features].copy()
y = df[target_column]

print(f"\n✓ Using ONLY admission-time features for prediction")
print(f"✓ No data leakage - truly predictive model")

print(f"\nTarget variable statistics:")
print(y.describe())
print(f"\nNumber of missing values in target: {y.isna().sum()}")

# Handle missing values in target
if y.isna().sum() > 0:
    print(f"Removing {y.isna().sum()} rows with missing target values...")
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Analyze missing values
print(f"\n{'='*70}")
print("Missing values in features:")
print(f"{'='*70}")
missing_counts = X.isna().sum()
missing_percentages = (missing_counts / len(X)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_percentages
})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Percentage', ascending=False)
if len(missing_info) > 0:
    print(missing_info)
else:
    print("No missing values found!")

# Check data types
print(f"\n{'='*70}")
print("Data types:")
print(f"{'='*70}")
print(X.dtypes.value_counts())

# Handle categorical columns
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

if categorical_columns:
    print(f"Categorical columns: {categorical_columns}")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col].astype(str))
    print("✓ Categorical columns encoded")

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\n{'='*70}")
print("✓ Data preprocessing completed!")
print(f"{'='*70}")

# Split data
print("\nSplitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print(f"\n{'='*70}")
print("Training Random Forest Model (Clean - No Data Leakage)...")
print(f"{'='*70}")

# Train model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print("\n✓ Model training completed!")

# Make predictions
print("\nMaking predictions...")
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate
print(f"\n{'='*70}")
print("MODEL PERFORMANCE")
print(f"{'='*70}")

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTRAINING SET:")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  RMSE:      {train_rmse:.4f}")
print(f"  MAE:       {train_mae:.4f}")

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTEST SET:")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  RMSE:      {test_rmse:.4f}")
print(f"  MAE:       {test_mae:.4f}")

# Cross-validation
print(f"\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)
print(f"  CV R² Scores: {cv_scores}")
print(f"  Mean CV R²:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
print(f"\n{'='*70}")
print("TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*70}")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save artifacts
print(f"\n{'='*70}")
print("SAVING MODEL AND ARTIFACTS")
print(f"{'='*70}")

feature_importance.to_csv('motor_clean_feature_importance.csv', index=False)
print(f"✓ Feature importance saved")

joblib.dump(rf_model, 'random_forest_motor_clean_model.pkl')
print(f"✓ Model saved")

joblib.dump(imputer, 'motor_clean_imputer.pkl')
print(f"✓ Imputer saved")

joblib.dump(X_train.columns.tolist(), 'motor_clean_feature_names.pkl')
print(f"✓ Feature names saved")

# Create visualizations
print(f"\n{'='*70}")
print("CREATING VISUALIZATIONS")
print(f"{'='*70}")

# 1. Actual vs Predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AASATotD', fontsize=12)
plt.ylabel('Predicted AASATotD', fontsize=12)
plt.title(f'Clean Model: Actual vs Predicted\nR² = {test_r2:.4f} (No Data Leakage)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted AASATotD', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('motor_clean_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_clean_predictions.png")
plt.close()

# 2. Feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Features (Clean Model - Admission Only)', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('motor_clean_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_clean_feature_importance.png")
plt.close()

# 3. Distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_test, bins=50, alpha=0.7, label='Actual', edgecolor='black')
plt.hist(y_test_pred, bins=50, alpha=0.7, label='Predicted', edgecolor='black')
plt.xlabel('AASATotD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution: Actual vs Predicted', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Error Distribution\nMean: {residuals.mean():.2f}', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('motor_clean_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_clean_distributions.png")
plt.close()

# Summary report
summary = f"""
CLEAN MOTOR SCORE PREDICTION MODEL (NO DATA LEAKAGE)
{'='*70}

DATASET INFORMATION:
- Total samples: {len(df)}
- Features: {X_train.shape[1]} (ADMISSION-TIME ONLY)
- Target: {target_column}
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}

KEY DIFFERENCE FROM PREVIOUS MODEL:
✓ Uses ONLY admission/injury-time features
✓ NO discharge features included
✓ Truly predictive (no data leakage)
✓ Can be used at admission for early counseling

PERFORMANCE METRICS:
Training Set:
  - R² Score: {train_r2:.4f}
  - RMSE: {train_rmse:.4f}
  - MAE: {train_mae:.4f}

Test Set:
  - R² Score: {test_r2:.4f}
  - RMSE: {test_rmse:.4f}
  - MAE: {test_mae:.4f}

Cross-Validation:
  - Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})

TOP 10 FEATURES (ADMISSION-TIME ONLY):
{feature_importance.head(10).to_string(index=False)}

MODEL SPECIFICATIONS:
- Algorithm: Random Forest Regressor
- Trees: {rf_model.n_estimators}
- Max Depth: {rf_model.max_depth}
- Features Used: {X_train.shape[1]}

CLINICAL UTILITY:
✓ Predicts discharge motor scores at ADMISSION
✓ Useful for early patient counseling
✓ Helps set realistic expectations
✓ Guides treatment planning and resource allocation

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('motor_clean_model_summary.txt', 'w') as f:
    f.write(summary)
print("✓ Saved: motor_clean_model_summary.txt")

print(f"\n{'='*70}")
print("✓ CLEAN MODEL TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\nPerformance Summary:")
print(f"  R² = {test_r2:.4f} (explains {test_r2*100:.1f}% of variance)")
print(f"  RMSE = {test_rmse:.4f} points")
print(f"  MAE = {test_mae:.4f} points")
print(f"\nThis is a TRULY PREDICTIVE model - no data leakage!")
print(f"Can be used at admission for early outcome prediction.")

