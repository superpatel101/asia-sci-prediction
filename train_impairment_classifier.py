import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
# Load the data
df = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Separate features and target
target_column = 'AASAImDs'
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
print(f"\nTarget value counts:")
print(y.value_counts().sort_index())
print(f"\nNumber of missing values in target: {y.isna().sum()}")

# Handle missing values in target (drop rows with missing target)
if y.isna().sum() > 0:
    print(f"Removing {y.isna().sum()} rows with missing target values...")
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    print(f"New shape: {X.shape}")

# Check if target is numeric or categorical
print(f"\nTarget data type: {y.dtype}")
print(f"Unique target values: {sorted(y.unique())}")

# Convert to integer if needed
y = y.astype(int)

# Map numeric labels to ASIA grades for interpretation
asia_grade_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
print(f"\nASIA Impairment Grade Mapping:")
for num, grade in asia_grade_map.items():
    count = (y == num).sum()
    pct = (count / len(y)) * 100
    print(f"  {num} (Grade {grade}): {count:5d} samples ({pct:5.2f}%)")

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
if len(missing_info) > 0:
    print(missing_info)
else:
    print("No missing values found!")

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
    # Convert categorical to numeric (label encoding)
    le = LabelEncoder()
    for col in categorical_columns:
        X[col] = le.fit_transform(X[col].astype(str))
    print("Categorical columns have been label encoded.")

# Handle missing values in features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"\n{'='*60}")
print("Data preprocessing completed!")
print(f"{'='*60}")

# Check class balance
print(f"\nClass Balance Analysis:")
class_counts = y.value_counts().sort_index()
for cls in class_counts.index:
    pct = (class_counts[cls] / len(y)) * 100
    print(f"  Class {cls} (Grade {asia_grade_map.get(cls, '?')}): {class_counts[cls]:5d} ({pct:5.2f}%)")

# Split the data into training and testing sets
print("\nSplitting data into train (80%) and test (20%) sets (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Verify stratification
print("\nTrain set class distribution:")
print(y_train.value_counts().sort_index())
print("\nTest set class distribution:")
print(y_test.value_counts().sort_index())

print(f"\n{'='*60}")
print("Training Random Forest Classifier...")
print(f"{'='*60}")

# Initialize Random Forest Classifier with good default parameters
rf_model = RandomForestClassifier(
    n_estimators=200,          # Number of trees
    max_depth=20,              # Maximum depth of trees
    min_samples_split=5,       # Minimum samples required to split a node
    min_samples_leaf=2,        # Minimum samples required at a leaf node
    max_features='sqrt',       # Number of features to consider for best split
    class_weight='balanced',   # Handle class imbalance
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

# Get probability predictions for AUC calculation
y_train_proba = rf_model.predict_proba(X_train)
y_test_proba = rf_model.predict_proba(X_test)

# Evaluate the model
print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

print(f"\nTRAINING SET METRICS:")
print(f"  Accuracy:           {train_accuracy:.4f}")
print(f"  F1-Score (Macro):   {train_f1_macro:.4f}")
print(f"  F1-Score (Weighted):{train_f1_weighted:.4f}")

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"\nTEST SET METRICS:")
print(f"  Accuracy:           {test_accuracy:.4f}")
print(f"  F1-Score (Macro):   {test_f1_macro:.4f}")
print(f"  F1-Score (Weighted):{test_f1_weighted:.4f}")
print(f"  Precision:          {test_precision:.4f}")
print(f"  Recall:             {test_recall:.4f}")

# Calculate multi-class AUC
try:
    from sklearn.preprocessing import label_binarize
    classes = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    if len(classes) == 2:
        test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    else:
        test_auc = roc_auc_score(y_test_bin, y_test_proba, multi_class='ovr', average='weighted')
    print(f"  AUC (Weighted):     {test_auc:.4f}")
except Exception as e:
    print(f"  AUC calculation skipped: {e}")
    test_auc = None

# Detailed classification report
print(f"\n{'='*60}")
print("DETAILED CLASSIFICATION REPORT (TEST SET)")
print(f"{'='*60}")
print(classification_report(y_test, y_test_pred, 
                          target_names=[f"Grade {asia_grade_map.get(c, c)}" for c in sorted(y_test.unique())]))

# Confusion Matrix
print(f"\n{'='*60}")
print("CONFUSION MATRIX (TEST SET)")
print(f"{'='*60}")
cm = confusion_matrix(y_test, y_test_pred)
print("\nRows = Actual, Columns = Predicted")
print(f"Classes: {[asia_grade_map.get(c, c) for c in sorted(y_test.unique())]}")
print(cm)

# Cross-validation
print(f"\nPerforming 5-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=skf, 
                            scoring='accuracy', n_jobs=-1)
print(f"  CV Accuracy Scores: {cv_scores}")
print(f"  Mean CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

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
feature_importance.to_csv('impairment_feature_importance.csv', index=False)
print("\n✓ Feature importance saved to 'impairment_feature_importance.csv'")

# Save the model
print(f"\n{'='*60}")
print("SAVING MODEL AND ARTIFACTS")
print(f"{'='*60}")

model_filename = 'random_forest_impairment_classifier.pkl'
joblib.dump(rf_model, model_filename)
print(f"✓ Model saved to '{model_filename}'")

# Save the imputer for future predictions
imputer_filename = 'impairment_imputer.pkl'
joblib.dump(imputer, imputer_filename)
print(f"✓ Imputer saved to '{imputer_filename}'")

# Save feature names
feature_names_filename = 'impairment_feature_names.pkl'
joblib.dump(X_train.columns.tolist(), feature_names_filename)
print(f"✓ Feature names saved to '{feature_names_filename}'")

# Create visualizations
print(f"\n{'='*60}")
print("CREATING VISUALIZATIONS")
print(f"{'='*60}")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix\n(ASIA Impairment at Discharge)', fontsize=14)
plt.colorbar()
tick_marks = np.arange(len(sorted(y_test.unique())))
labels = [f"Grade {asia_grade_map.get(c, c)}" for c in sorted(y_test.unique())]
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

# Add text annotations
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=10)

plt.ylabel('Actual Grade', fontsize=12)
plt.xlabel('Predicted Grade', fontsize=12)
plt.tight_layout()
plt.savefig('impairment_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved to 'impairment_confusion_matrix.png'")
plt.close()

# 2. Feature importance plot
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Most Important Features', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('impairment_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved to 'impairment_feature_importance.png'")
plt.close()

# 3. Class distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual distribution
axes[0].bar([asia_grade_map.get(c, c) for c in sorted(y_test.unique())], 
           y_test.value_counts().sort_index().values, 
           alpha=0.7, edgecolor='black')
axes[0].set_xlabel('ASIA Grade', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Test Set: Actual Distribution', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Predicted distribution
pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
axes[1].bar([asia_grade_map.get(c, c) for c in sorted(pred_counts.index)], 
           pred_counts.values, 
           alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('ASIA Grade', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Test Set: Predicted Distribution', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('impairment_class_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Class distribution plot saved to 'impairment_class_distributions.png'")
plt.close()

# 4. Per-class performance
per_class_report = classification_report(y_test, y_test_pred, output_dict=True)
classes_in_test = sorted(y_test.unique())
class_names = [asia_grade_map.get(c, str(c)) for c in classes_in_test]
f1_scores = [per_class_report[str(c)]['f1-score'] for c in classes_in_test]
precision_scores = [per_class_report[str(c)]['precision'] for c in classes_in_test]
recall_scores = [per_class_report[str(c)]['recall'] for c in classes_in_test]

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('ASIA Grade', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f"Grade {c}" for c in class_names])
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('impairment_per_class_performance.png', dpi=300, bbox_inches='tight')
print("✓ Per-class performance plot saved to 'impairment_per_class_performance.png'")
plt.close()

# Create a summary report
print(f"\n{'='*60}")
print("CREATING SUMMARY REPORT")
print(f"{'='*60}")

summary_report = f"""
RANDOM FOREST CLASSIFIER - ASIA IMPAIRMENT PREDICTION
{'='*60}

DATASET INFORMATION:
- Total samples: {len(df)}
- Features: {X_train.shape[1]}
- Target variable: {target_column} (ASIA Impairment at Discharge)
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
- Number of classes: {len(y.unique())}
- Classes: {[f"{k} (Grade {v})" for k, v in asia_grade_map.items() if k in y.unique()]}

CLASS DISTRIBUTION:
{y.value_counts().sort_index().to_string()}

MODEL PARAMETERS:
- Algorithm: Random Forest Classifier
- Number of trees: {rf_model.n_estimators}
- Max depth: {rf_model.max_depth}
- Min samples split: {rf_model.min_samples_split}
- Min samples leaf: {rf_model.min_samples_leaf}
- Max features: {rf_model.max_features}
- Class weight: {rf_model.class_weight}

PERFORMANCE METRICS:
Training Set:
  - Accuracy: {train_accuracy:.4f}
  - F1-Score (Macro): {train_f1_macro:.4f}
  - F1-Score (Weighted): {train_f1_weighted:.4f}

Test Set:
  - Accuracy: {test_accuracy:.4f}
  - F1-Score (Macro): {test_f1_macro:.4f}
  - F1-Score (Weighted): {test_f1_weighted:.4f}
  - Precision: {test_precision:.4f}
  - Recall: {test_recall:.4f}
  {"- AUC (Weighted): " + f"{test_auc:.4f}" if test_auc else ""}

Cross-Validation:
  - Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})

CONFUSION MATRIX (Test Set):
{cm}

TOP 10 MOST IMPORTANT FEATURES:
{feature_importance.head(10).to_string(index=False)}

FILES GENERATED:
1. random_forest_impairment_classifier.pkl - Trained classifier
2. impairment_imputer.pkl - Data imputer for preprocessing
3. impairment_feature_names.pkl - List of feature names
4. impairment_feature_importance.csv - Complete feature importance rankings
5. impairment_confusion_matrix.png - Confusion matrix heatmap
6. impairment_feature_importance.png - Feature importance visualization
7. impairment_class_distributions.png - Class distribution comparison
8. impairment_per_class_performance.png - Per-class metrics
9. impairment_model_summary.txt - This summary report

MODEL INTERPRETATION:
- Accuracy indicates overall correct classification rate
- F1-Score balances precision and recall
- Macro F1 treats all classes equally
- Weighted F1 accounts for class imbalance
- Feature importance shows which variables most influence predictions

{'='*60}
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('impairment_model_summary.txt', 'w') as f:
    f.write(summary_report)

print("✓ Summary report saved to 'impairment_model_summary.txt'")

print(f"\n{'='*60}")
print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print("\nTo use the model for predictions on new data:")
print("  1. Load the model: model = joblib.load('random_forest_impairment_classifier.pkl')")
print("  2. Load the imputer: imputer = joblib.load('impairment_imputer.pkl')")
print("  3. Preprocess new data using the imputer")
print("  4. Make predictions: predictions = model.predict(new_data)")
print("  5. Get probabilities: probabilities = model.predict_proba(new_data)")
print(f"\n{'='*60}\n")

