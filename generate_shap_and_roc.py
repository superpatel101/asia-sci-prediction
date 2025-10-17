"""
Generate SHAP (SHapley Additive exPlanations) plots for both models
and create a comprehensive PDF report for research publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GENERATING SHAP ANALYSIS FOR BOTH MODELS")
print("="*70)

# Check if shap is installed, if not, install it
try:
    import shap
    print("✓ SHAP library loaded")
except ImportError:
    print("Installing SHAP library...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'shap'])
    import shap
    print("✓ SHAP library installed and loaded")

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# ============================================================================
# MODEL 1: MOTOR SCORE REGRESSION - SHAP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("MODEL 1: MOTOR SCORE REGRESSION - SHAP ANALYSIS")
print("="*70)

# Load Model 1
print("\nLoading Model 1 artifacts...")
model1 = joblib.load('random_forest_asia_motor_model.pkl')
imputer1 = joblib.load('imputer.pkl')
feature_names1 = joblib.load('feature_names.pkl')

# Load data
print("Loading Model 1 data...")
df1 = pd.read_csv('V2_EDIT_modelreadyASIAMotor.csv')
X1 = df1.drop(columns=['AASATotD'])
y1 = df1['AASATotD']

# Preprocess
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['AInjAge', 'AASAImAd', 'AASAImDs', 'ANurLvlA', 'ANurLvlD']
for col in categorical_columns:
    if col in X1.columns and X1[col].dtype == 'object':
        le = LabelEncoder()
        X1[col] = le.fit_transform(X1[col].astype(str))

X1_processed = pd.DataFrame(imputer1.transform(X1), columns=feature_names1)

# Sample for SHAP (use subset for speed)
print("Sampling data for SHAP analysis...")
sample_size = min(1000, len(X1_processed))
X1_sample = X1_processed.sample(n=sample_size, random_state=42)

# Calculate SHAP values
print("Computing SHAP values for Model 1 (this may take a few minutes)...")
explainer1 = shap.TreeExplainer(model1)
shap_values1 = explainer1.shap_values(X1_sample)

print("✓ SHAP values computed for Model 1")

# Create SHAP summary plots
print("Creating SHAP visualizations for Model 1...")

# Beeswarm plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values1, X1_sample, show=False, max_display=20)
plt.title('SHAP Summary Plot - Model 1: Motor Score Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary_model1_motor.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_summary_model1_motor.png")
plt.close()

# Bar plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values1, X1_sample, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance - Model 1: Motor Score Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_bar_model1_motor.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_bar_model1_motor.png")
plt.close()

# ============================================================================
# MODEL 2: IMPAIRMENT CLASSIFICATION - SHAP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("MODEL 2: IMPAIRMENT CLASSIFICATION - SHAP ANALYSIS")
print("="*70)

# Load Model 2
print("\nLoading Model 2 artifacts...")
model2 = joblib.load('random_forest_impairment_classifier.pkl')
imputer2 = joblib.load('impairment_imputer.pkl')
feature_names2 = joblib.load('impairment_feature_names.pkl')

# Load data
print("Loading Model 2 data...")
df2 = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')
X2 = df2.drop(columns=['AASAImDs'])
y2 = df2['AASAImDs'].astype(int)

# Preprocess
categorical_columns2 = ['AInjAge', 'AASAImAd', 'ANurLvlA']
for col in categorical_columns2:
    if col in X2.columns and X2[col].dtype == 'object':
        le = LabelEncoder()
        X2[col] = le.fit_transform(X2[col].astype(str))

X2_processed = pd.DataFrame(imputer2.transform(X2), columns=feature_names2)

# Sample for SHAP
print("Sampling data for SHAP analysis...")
sample_size2 = min(1000, len(X2_processed))
X2_sample = X2_processed.sample(n=sample_size2, random_state=42)
y2_sample = y2.iloc[X2_sample.index]

# Calculate SHAP values (for multi-class, we get values for each class)
print("Computing SHAP values for Model 2 (this may take a few minutes)...")
explainer2 = shap.TreeExplainer(model2)
shap_values2 = explainer2.shap_values(X2_sample)

print("✓ SHAP values computed for Model 2")

# Create SHAP summary plots
print("Creating SHAP visualizations for Model 2...")

# For multi-class, show the beeswarm for each class
ASIA_GRADE_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

# Overall summary (average across classes)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values2, X2_sample, show=False, max_display=20)
plt.title('SHAP Summary Plot - Model 2: Impairment Classification (All Classes)', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_summary_model2_impairment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_summary_model2_impairment.png")
plt.close()

# Bar plot (average importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values2, X2_sample, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance - Model 2: Impairment Classification', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('shap_bar_model2_impairment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_bar_model2_impairment.png")
plt.close()

# ============================================================================
# ROC CURVES
# ============================================================================

print("\n" + "="*70)
print("GENERATING ROC CURVES")
print("="*70)

# ROC for Model 2 (Classification)
print("\nGenerating ROC curves for Model 2...")
from sklearn.model_selection import train_test_split

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_processed, y2, test_size=0.2, random_state=42, stratify=y2
)

# Get predictions
y2_pred_proba = model2.predict_proba(X2_test)

# Binarize the output for multi-class ROC
classes = sorted(y2.unique())
y2_test_bin = label_binarize(y2_test, classes=classes)
n_classes = len(classes)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, class_val in enumerate(classes):
    fpr[i] = []
    tpr[i] = []
    roc_auc[i] = 0
    
    if class_val in y2_test.values:
        fpr[i], tpr[i], _ = roc_curve(y2_test_bin[:, i], y2_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, (class_val, color) in enumerate(zip(classes, colors)):
    if roc_auc[i] > 0:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Grade {ASIA_GRADE_MAP[class_val]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model 2: ASIA Impairment Classification', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_model2_impairment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves_model2_impairment.png")
plt.close()

# Compute micro-average and macro-average ROC curve
from sklearn.metrics import roc_auc_score

# Micro-average
fpr_micro, tpr_micro, _ = roc_curve(y2_test_bin.ravel(), y2_pred_proba.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average (simple average of individual AUCs)
roc_auc_macro = np.mean([roc_auc[i] for i in range(n_classes) if roc_auc[i] > 0])

# Weighted average (using class frequencies)
roc_auc_weighted = roc_auc_score(y2_test_bin, y2_pred_proba, average='weighted', multi_class='ovr')

print(f"\nROC AUC Scores for Model 2:")
print(f"  Micro-average: {roc_auc_micro:.4f}")
print(f"  Macro-average: {roc_auc_macro:.4f}")
print(f"  Weighted-average: {roc_auc_weighted:.4f}")

# Enhanced ROC plot with micro and macro averages
plt.figure(figsize=(12, 9))

# Plot individual class ROC curves
for i, (class_val, color) in enumerate(zip(classes, colors)):
    if roc_auc[i] > 0:
        plt.plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.6,
                label=f'Grade {ASIA_GRADE_MAP[class_val]} (AUC = {roc_auc[i]:.3f})')

# Plot micro-average
plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=3,
         label=f'Micro-average (AUC = {roc_auc_micro:.3f})')

# Plot macro-average (approximate)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curves with Averages - Model 2: ASIA Impairment Classification', fontsize=14, pad=15)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_enhanced_model2.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves_enhanced_model2.png")
plt.close()

print("\n" + "="*70)
print("✓ ALL SHAP AND ROC VISUALIZATIONS GENERATED!")
print("="*70)

# Save AUC statistics
auc_stats = {
    'Model 2 - Impairment Classification': {
        'Per-Class AUC': {f'Grade {ASIA_GRADE_MAP[classes[i]]}': roc_auc[i] for i in range(n_classes) if roc_auc[i] > 0},
        'Micro-Average AUC': float(roc_auc_micro),
        'Macro-Average AUC': float(roc_auc_macro),
        'Weighted-Average AUC': float(roc_auc_weighted)
    }
}

import json
with open('auc_statistics.json', 'w') as f:
    json.dump(auc_stats, f, indent=2)
print("\n✓ AUC statistics saved to 'auc_statistics.json'")

print("\nGenerated files:")
print("  1. shap_summary_model1_motor.png")
print("  2. shap_bar_model1_motor.png")
print("  3. shap_summary_model2_impairment.png")
print("  4. shap_bar_model2_impairment.png")
print("  5. roc_curves_model2_impairment.png")
print("  6. roc_curves_enhanced_model2.png")
print("  7. auc_statistics.json")

