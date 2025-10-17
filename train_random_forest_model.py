import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
# Load the data
df = pd.read_csv('V2_EDIT_modelreadyASIAMotor.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Separate features and target
target_column = 'AASATotD'
print(f"\n{'='*60}")
print(f"Target variable: {target_column}")
print(f"{'='*60}")

# Check if target exists
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset!")

X = df.drop(columns=[target_column])
y = df[target_column]

print(f"\nTarget variable statistics:")
print(y.describe())
print(f"\nNumber of missing values in target: {y.isna().sum()}")

# Handle missing values in target (drop rows with missing target)
if y.isna().sum() > 0:
    print(f"Removing {y.isna().sum()} rows with missing target values...")
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Analyze and handle missing values in features
print(f"\n{'='*60}")
print("Missing values in features:")
print(f"{'='*60}")
missing_counts = X.isna().sum()
missing_percentages = (missing_counts / len(X)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_percentages
})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Percentage', ascending=False)
print(missing_info)

# Check data types
print(f"\n{'='*60}")
print("Data types of features:")
print(f"{'='*60}")
print(X.dtypes.value_counts())

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

if categorical_columns:
    print(f"\nCategorical columns detected: {categorical_columns}")
    # Convert categorical to numeric (one-hot encoding or label encoding)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col].astype(str))
    print("Categorical columns have been label encoded.")

# Handle missing values in features
# For simplicity, we'll use median imputation for numeric columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\n{'='*60}")
print("Data preprocessing completed!")
print(f"{'='*60}")

# Split the data into training and testing sets
print("\nSplitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Feature scaling (optional for Random Forest, but can help with interpretation)
# Commenting out as RF doesn't require scaling, but keeping code for reference
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

print(f"\n{'='*60}")
print("Training Random Forest Model...")
print(f"{'='*60}")

# Initialize Random Forest Regressor with good default parameters
rf_model = RandomForestRegressor(
    n_estimators=200,          # Number of trees
    max_depth=20,              # Maximum depth of trees
    min_samples_split=5,       # Minimum samples required to split a node
    min_samples_leaf=2,        # Minimum samples required at a leaf node
    max_features='sqrt',       # Number of features to consider for best split
    random_state=42,
    n_jobs=-1,                 # Use all available cores
    verbose=1
)

# Train the model
rf_model.fit(X_train, y_train)

print("\n✓ Model training completed!")

# Make predictions
print("\nMaking predictions...")
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate the model
print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTRAINING SET METRICS:")
print(f"  R² Score:           {train_r2:.4f}")
print(f"  RMSE:               {train_rmse:.4f}")
print(f"  MAE:                {train_mae:.4f}")
print(f"  MSE:                {train_mse:.4f}")

# Test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTEST SET METRICS:")
print(f"  R² Score:           {test_r2:.4f}")
print(f"  RMSE:               {test_rmse:.4f}")
print(f"  MAE:                {test_mae:.4f}")
print(f"  MSE:                {test_mse:.4f}")

# Cross-validation
print(f"\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)
print(f"  CV R² Scores:       {cv_scores}")
print(f"  Mean CV R² Score:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
print(f"\n{'='*60}")
print("TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("\n✓ Feature importance saved to 'feature_importance.csv'")

# Save the model
print(f"\n{'='*60}")
print("SAVING MODEL AND ARTIFACTS")
print(f"{'='*60}")

model_filename = 'random_forest_asia_motor_model.pkl'
joblib.dump(rf_model, model_filename)
print(f"✓ Model saved to '{model_filename}'")

# Save the imputer for future predictions
imputer_filename = 'imputer.pkl'
joblib.dump(imputer, imputer_filename)
print(f"✓ Imputer saved to '{imputer_filename}'")

# Save feature names
feature_names_filename = 'feature_names.pkl'
joblib.dump(X_train.columns.tolist(), feature_names_filename)
print(f"✓ Feature names saved to '{feature_names_filename}'")

# Create visualizations
print(f"\n{'='*60}")
print("CREATING VISUALIZATIONS")
print(f"{'='*60}")

# 1. Actual vs Predicted plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AASATotD', fontsize=12)
plt.ylabel('Predicted AASATotD', fontsize=12)
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}', fontsize=14)
plt.grid(True, alpha=0.3)

# 2. Residuals plot
residuals = y_test - y_test_pred
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted AASATotD', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Prediction plot saved to 'model_predictions.png'")
plt.close()

# 3. Feature importance plot
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Most Important Features', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved to 'feature_importance.png'")
plt.close()

# 4. Distribution comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_test, bins=50, alpha=0.7, label='Actual', edgecolor='black')
plt.hist(y_test_pred, bins=50, alpha=0.7, label='Predicted', edgecolor='black')
plt.xlabel('AASATotD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution: Actual vs Predicted', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Error distribution
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Error Distribution\nMean Error: {residuals.mean():.4f}', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
print("✓ Distribution plots saved to 'distributions.png'")
plt.close()

# Create a summary report
print(f"\n{'='*60}")
print("CREATING SUMMARY REPORT")
print(f"{'='*60}")

summary_report = f"""
RANDOM FOREST MODEL - ASIA MOTOR SCORE PREDICTION
{'='*60}

DATASET INFORMATION:
- Total samples: {len(df)}
- Features: {X_train.shape[1]}
- Target variable: {target_column}
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}

MODEL PARAMETERS:
- Algorithm: Random Forest Regressor
- Number of trees: {rf_model.n_estimators}
- Max depth: {rf_model.max_depth}
- Min samples split: {rf_model.min_samples_split}
- Min samples leaf: {rf_model.min_samples_leaf}
- Max features: {rf_model.max_features}

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
  - Mean R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})

TOP 10 MOST IMPORTANT FEATURES:
{feature_importance.head(10).to_string(index=False)}

FILES GENERATED:
1. random_forest_asia_motor_model.pkl - Trained model
2. imputer.pkl - Data imputer for preprocessing
3. feature_names.pkl - List of feature names
4. feature_importance.csv - Complete feature importance rankings
5. model_predictions.png - Actual vs predicted & residual plots
6. feature_importance.png - Feature importance visualization
7. distributions.png - Distribution comparison plots
8. model_summary_report.txt - This summary report

MODEL INTERPRETATION:
- R² Score indicates the proportion of variance explained by the model
- RMSE (Root Mean Squared Error) represents average prediction error
- MAE (Mean Absolute Error) represents average absolute prediction error
- Feature importance shows which variables most influence predictions

{'='*60}
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('model_summary_report.txt', 'w') as f:
    f.write(summary_report)

print("✓ Summary report saved to 'model_summary_report.txt'")

print(f"\n{'='*60}")
print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print("\nTo use the model for predictions on new data:")
print("  1. Load the model: model = joblib.load('random_forest_asia_motor_model.pkl')")
print("  2. Load the imputer: imputer = joblib.load('imputer.pkl')")
print("  3. Preprocess new data using the imputer")
print("  4. Make predictions: predictions = model.predict(new_data)")
print(f"\n{'='*60}\n")

