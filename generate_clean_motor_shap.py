"""
Generate SHAP plots for the CLEAN motor score model (no data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GENERATING SHAP ANALYSIS FOR CLEAN MOTOR SCORE MODEL")
print("="*70)

import shap

# Load clean model
print("\nLoading clean model artifacts...")
model = joblib.load('random_forest_motor_clean_model.pkl')
imputer = joblib.load('motor_clean_imputer.pkl')
feature_names = joblib.load('motor_clean_feature_names.pkl')

# Load data
print("Loading data...")
df = pd.read_csv('/Users/aaryanpatel/Downloads/V2_EDIT_modelreadyASIAMotor.csv')

# Get admission features only
admission_features = feature_names
X = df[admission_features].copy()
y = df['AASATotD']

# Preprocess
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['AInjAge', 'AASAImAd', 'ANurLvlA']
for col in categorical_columns:
    if col in X.columns and X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X_processed = pd.DataFrame(imputer.transform(X), columns=feature_names)

# Sample for SHAP
print("Sampling data for SHAP analysis...")
sample_size = min(1000, len(X_processed))
X_sample = X_processed.sample(n=sample_size, random_state=42)

# Calculate SHAP values
print("Computing SHAP values (this may take a few minutes)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("✓ SHAP values computed")

# Create SHAP visualizations
print("Creating SHAP visualizations...")

# Beeswarm plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.title('SHAP Summary - Clean Motor Score Model\n(Admission Features Only - No Data Leakage)', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary_motor_clean.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_summary_motor_clean.png")
plt.close()

# Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance - Clean Motor Score Model\n(Truly Predictive)', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_bar_motor_clean.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_bar_motor_clean.png")
plt.close()

print("\n✓ SHAP analysis complete for clean motor model!")

