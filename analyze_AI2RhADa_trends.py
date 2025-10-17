"""
Analyze the relationship between AI2RhADa (days from injury to rehab admission)
and ASIA impairment grades to understand directional trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALYZING AI2RhADa (Days to Rehab) vs ASIA IMPAIRMENT GRADES")
print("="*80)

# Load the model and preprocessing artifacts
print("\nLoading model and artifacts...")
model = joblib.load('random_forest_impairment_classifier.pkl')
imputer = joblib.load('impairment_imputer.pkl')
feature_names = joblib.load('impairment_feature_names.pkl')

# Load data
print("Loading data...")
df = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')
X = df.drop(columns=['AASAImDs'])
y = df['AASAImDs'].astype(int)

# Preprocess
categorical_columns = ['AInjAge', 'AASAImAd', 'ANurLvlA']
for col in categorical_columns:
    if col in X.columns and X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X_processed = pd.DataFrame(imputer.transform(X), columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Get the original AI2RhADa values (before imputation) for interpretation
AI2RhADa_original = df['AI2RhADa'].copy()
AI2RhADa_train_original = AI2RhADa_original.iloc[X_train.index]
AI2RhADa_test_original = AI2RhADa_original.iloc[X_test.index]

ASIA_GRADE_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

print(f"\nDataset info:")
print(f"  Total samples: {len(df)}")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  AI2RhADa feature index: {feature_names.index('AI2RhADa')}")

# ============================================================================
# PART 1: DESCRIPTIVE STATISTICS - Actual trends in the data
# ============================================================================

print("\n" + "="*80)
print("PART 1: DESCRIPTIVE STATISTICS")
print("="*80)

print("\nAI2RhADa statistics by ASIA grade (actual data):")
print("-" * 80)

stats_by_grade = []
for grade_num in sorted(y.unique()):
    grade_letter = ASIA_GRADE_MAP[grade_num]
    mask = y == grade_num
    values = AI2RhADa_original[mask]
    
    stats = {
        'Grade': f"{grade_num} ({grade_letter})",
        'Count': len(values),
        'Mean': values.mean(),
        'Median': values.median(),
        'Std': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'Q25': values.quantile(0.25),
        'Q75': values.quantile(0.75)
    }
    stats_by_grade.append(stats)
    
    print(f"\nGrade {grade_num} ({grade_letter}): {len(values)} patients")
    print(f"  Mean:   {values.mean():.1f} days")
    print(f"  Median: {values.median():.1f} days")
    print(f"  Std:    {values.std():.1f} days")
    print(f"  Range:  {values.min():.0f} - {values.max():.0f} days")
    print(f"  IQR:    {values.quantile(0.25):.1f} - {values.quantile(0.75):.1f} days")

stats_df = pd.DataFrame(stats_by_grade)
print("\n" + "="*80)
print("Summary Table:")
print(stats_df.to_string(index=False))

# ============================================================================
# PART 2: VISUALIZATIONS - Show the actual trends
# ============================================================================

print("\n" + "="*80)
print("PART 2: CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Distribution by grade
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('AI2RhADa (Days from Injury to Rehab) Distribution by ASIA Grade', 
             fontsize=16, fontweight='bold')

for idx, grade_num in enumerate(sorted(y.unique())):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    grade_letter = ASIA_GRADE_MAP[grade_num]
    mask = y == grade_num
    values = AI2RhADa_original[mask]
    
    ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.1f}')
    ax.axvline(values.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {values.median():.1f}')
    ax.set_xlabel('Days from Injury to Rehab', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Grade {grade_num} ({grade_letter}) - n={len(values)}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove empty subplot if odd number of grades
if len(sorted(y.unique())) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('AI2RhADa_distributions_by_grade.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_distributions_by_grade.png")
plt.close()

# Figure 2: Box plot comparison
plt.figure(figsize=(12, 7))
data_for_box = [AI2RhADa_original[y == grade_num] for grade_num in sorted(y.unique())]
labels_for_box = [f"Grade {grade_num}\n({ASIA_GRADE_MAP[grade_num]})" for grade_num in sorted(y.unique())]

bp = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                  notch=True, showmeans=True)

# Color the boxes
colors = ['#ff6b6b', '#ee5a6f', '#4ecdc4', '#45b7d1', '#96ceb4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('Days from Injury to Rehab Admission (AI2RhADa)', fontsize=13)
plt.xlabel('ASIA Impairment Grade', fontsize=13)
plt.title('AI2RhADa Distribution Across ASIA Grades\n(Box = IQR, Notch = 95% CI of median, Diamond = mean)',
          fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('AI2RhADa_boxplot_by_grade.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_boxplot_by_grade.png")
plt.close()

# Figure 3: Mean comparison with error bars
plt.figure(figsize=(12, 7))
means = [AI2RhADa_original[y == g].mean() for g in sorted(y.unique())]
stds = [AI2RhADa_original[y == g].std() for g in sorted(y.unique())]
counts = [len(AI2RhADa_original[y == g]) for g in sorted(y.unique())]
sems = [std / np.sqrt(count) for std, count in zip(stds, counts)]

x_pos = np.arange(len(sorted(y.unique())))
bars = plt.bar(x_pos, means, yerr=sems, alpha=0.7, capsize=10, 
               color=colors, edgecolor='black', linewidth=2)

plt.xticks(x_pos, [f"Grade {g}\n({ASIA_GRADE_MAP[g]})" for g in sorted(y.unique())])
plt.ylabel('Mean Days from Injury to Rehab (± SEM)', fontsize=13)
plt.xlabel('ASIA Impairment Grade', fontsize=13)
plt.title('Mean AI2RhADa by ASIA Grade\n(Error bars = Standard Error of Mean)', 
          fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
    plt.text(bar.get_x() + bar.get_width()/2, mean + sem + 2, 
             f'{mean:.1f}±{sem:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('AI2RhADa_mean_by_grade.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_mean_by_grade.png")
plt.close()

# ============================================================================
# PART 3: SHAP ANALYSIS - How the model uses AI2RhADa
# ============================================================================

print("\n" + "="*80)
print("PART 3: SHAP ANALYSIS")
print("="*80)

# Sample for SHAP (computational efficiency)
print("\nSampling data for SHAP analysis...")
sample_size = min(2000, len(X_test))
sample_indices = np.random.choice(X_test.index, size=sample_size, replace=False)
X_shap_sample = X_test.loc[sample_indices]
y_shap_sample = y_test.loc[sample_indices]
AI2RhADa_shap_sample = AI2RhADa_test_original.loc[sample_indices]

print(f"Computing SHAP values for {sample_size} test samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap_sample)

# SHAP values is a list (one array per class)
print(f"✓ SHAP values computed")
print(f"  Shape: {len(shap_values)} classes x {shap_values[0].shape}")

# Get the feature index for AI2RhADa
ai2rhada_idx = feature_names.index('AI2RhADa')

# Figure 4: SHAP dependency plot for AI2RhADa
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('SHAP Dependency Plot: AI2RhADa Impact on Each ASIA Grade Prediction', 
             fontsize=16, fontweight='bold')

for idx, grade_num in enumerate(sorted(y.unique())):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    grade_letter = ASIA_GRADE_MAP[grade_num]
    class_idx = list(model.classes_).index(grade_num)
    
    # Get SHAP values for this class
    shap_for_class = shap_values[class_idx][:, ai2rhada_idx]
    ai2rhada_values = X_shap_sample['AI2RhADa'].values
    
    # Create scatter plot
    scatter = ax.scatter(ai2rhada_values, shap_for_class, 
                        c=ai2rhada_values, cmap='coolwarm', 
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(ai2rhada_values, shap_for_class, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(ai2rhada_values.min(), ai2rhada_values.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('AI2RhADa (Days to Rehab)', fontsize=11)
    ax.set_ylabel(f'SHAP value\n(impact on Grade {grade_letter})', fontsize=11)
    ax.set_title(f'Grade {grade_num} ({grade_letter})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('AI2RhADa\nvalue', fontsize=9)

# Remove empty subplot if odd number of grades
if len(sorted(y.unique())) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('AI2RhADa_shap_dependency_by_grade.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_shap_dependency_by_grade.png")
plt.close()

# Figure 5: SHAP values summary for AI2RhADa across all classes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Average absolute SHAP value by class
ax = axes[0]
avg_abs_shap = []
for grade_num in sorted(y.unique()):
    class_idx = list(model.classes_).index(grade_num)
    avg_abs_shap.append(np.abs(shap_values[class_idx][:, ai2rhada_idx]).mean())

bars = ax.bar(range(len(sorted(y.unique()))), avg_abs_shap, 
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(sorted(y.unique()))))
ax.set_xticklabels([f"Grade {g}\n({ASIA_GRADE_MAP[g]})" for g in sorted(y.unique())])
ax.set_ylabel('Mean |SHAP| value', fontsize=12)
ax.set_xlabel('ASIA Impairment Grade', fontsize=12)
ax.set_title('Average AI2RhADa Impact Magnitude by Grade\n(How much does this feature matter?)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, avg_abs_shap)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel B: Mean SHAP value (showing direction)
ax = axes[1]
mean_shap = []
for grade_num in sorted(y.unique()):
    class_idx = list(model.classes_).index(grade_num)
    mean_shap.append(shap_values[class_idx][:, ai2rhada_idx].mean())

colors_directional = ['red' if x < 0 else 'green' for x in mean_shap]
bars = ax.bar(range(len(sorted(y.unique()))), mean_shap, 
              color=colors_directional, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(range(len(sorted(y.unique()))))
ax.set_xticklabels([f"Grade {g}\n({ASIA_GRADE_MAP[g]})" for g in sorted(y.unique())])
ax.set_ylabel('Mean SHAP value', fontsize=12)
ax.set_xlabel('ASIA Impairment Grade', fontsize=12)
ax.set_title('Average AI2RhADa Impact Direction by Grade\n(Positive = increases probability, Negative = decreases)', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, mean_shap)):
    y_pos = val + (0.001 if val > 0 else -0.001)
    va = 'bottom' if val > 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
            f'{val:.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('AI2RhADa_shap_summary_by_grade.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_shap_summary_by_grade.png")
plt.close()

# ============================================================================
# PART 4: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 4: STATISTICAL ANALYSIS")
print("="*80)

from scipy import stats

print("\nCorrelation between AI2RhADa and ASIA grade:")
correlation, p_value = stats.spearmanr(AI2RhADa_original, y)
print(f"  Spearman correlation: {correlation:.4f}")
print(f"  P-value: {p_value:.4e}")
print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'}")

# Pairwise comparisons
print("\nPairwise comparisons (Mann-Whitney U test):")
print("-" * 80)
grades = sorted(y.unique())
for i, grade1 in enumerate(grades):
    for grade2 in grades[i+1:]:
        group1 = AI2RhADa_original[y == grade1]
        group2 = AI2RhADa_original[y == grade2]
        stat, p = stats.mannwhitneyu(group1, group2)
        mean_diff = group1.mean() - group2.mean()
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"Grade {grade1} ({ASIA_GRADE_MAP[grade1]}) vs Grade {grade2} ({ASIA_GRADE_MAP[grade2]}): "
              f"Mean diff = {mean_diff:+.1f} days, p = {p:.4e} {sig}")

# ============================================================================
# PART 5: GENERATE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("PART 5: GENERATING SUMMARY REPORT")
print("="*80)

# Calculate correlation between AI2RhADa value and SHAP values for each class
correlations = {}
for grade_num in sorted(y.unique()):
    class_idx = list(model.classes_).index(grade_num)
    shap_for_class = shap_values[class_idx][:, ai2rhada_idx]
    ai2rhada_vals = X_shap_sample['AI2RhADa'].values
    corr, _ = stats.spearmanr(ai2rhada_vals, shap_for_class)
    correlations[grade_num] = corr

report = f"""
{'='*80}
ANALYSIS REPORT: AI2RhADa vs ASIA IMPAIRMENT GRADES
{'='*80}

VARIABLE: AI2RhADa (Days from Injury to Rehabilitation Admission)
MODEL: Random Forest Classifier for ASIA Impairment at Discharge

{'='*80}
QUESTION 1: What is the trend in AI2RhADa that correlates with each grade?
{'='*80}

ANSWER: The actual data shows the following pattern:

"""

for grade_num in sorted(y.unique()):
    mask = y == grade_num
    values = AI2RhADa_original[mask]
    report += f"\nGrade {grade_num} ({ASIA_GRADE_MAP[grade_num]}):\n"
    report += f"  - Mean: {values.mean():.1f} days (Median: {values.median():.1f})\n"
    report += f"  - Patients: {len(values)} ({len(values)/len(y)*100:.1f}% of dataset)\n"

report += f"""

KEY FINDING:
- Grade A (Complete injury): Mean = {AI2RhADa_original[y==1].mean():.1f} days
- Grade B (Sensory incomplete): Mean = {AI2RhADa_original[y==2].mean():.1f} days  
- Grade C (Motor incomplete <50%): Mean = {AI2RhADa_original[y==3].mean():.1f} days
- Grade D (Motor incomplete ≥50%): Mean = {AI2RhADa_original[y==4].mean():.1f} days
- Grade E (Normal): Mean = {AI2RhADa_original[y==5].mean():.1f} days

TREND INTERPRETATION:
"""

# Determine the trend
means_by_grade = {g: AI2RhADa_original[y==g].mean() for g in sorted(y.unique())}
if means_by_grade[1] < means_by_grade[5]:
    report += "More severe injuries (Grade A) tend to have SHORTER times to rehab admission,\n"
    report += "while less severe injuries (Grade E) have LONGER times to rehab admission.\n"
else:
    report += "More severe injuries (Grade A) tend to have LONGER times to rehab admission,\n"
    report += "while less severe injuries (Grade E) have SHORTER times to rehab admission.\n"

report += f"""

STATISTICAL SIGNIFICANCE:
- Spearman correlation with ASIA grade: {correlation:.4f} (p = {p_value:.4e})
- This correlation is {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}

{'='*80}
QUESTION 2: What does "7.5% feature importance" mean?
{'='*80}

ANSWER: The 7.5% Random Forest feature importance means:

1. DEFINITION:
   - When the Random Forest makes decisions (splits), it measures how much each
     feature improves the prediction accuracy
   - AI2RhADa contributes 7.5% to the total "improvement" across all trees
   - This is an AVERAGE measure across all classes and all patients

2. INTERPRETATION:
   - Out of 100% total importance distributed across all {len(feature_names)} features,
     AI2RhADa accounts for 7.5%
   - It is the {list(model.feature_importances_.argsort()[::-1]).index(feature_names.index('AI2RhADa')) + 1}th most important feature
   - It's relatively important but not dominant

3. WHAT IT DOESN'T TELL US:
   - It doesn't tell us the DIRECTION of the effect (positive or negative)
   - It doesn't tell us which CLASSES are affected more
   - It's a global average, not class-specific

{'='*80}
QUESTION 3: How does SHAP impact differ from feature importance?
{'='*80}

ANSWER: SHAP values provide much richer information:

1. FEATURE IMPORTANCE (7.5%):
   - Global: One number for the entire model
   - Direction-agnostic: Doesn't tell you if higher values help or hurt
   - Class-agnostic: Doesn't tell you which classes are affected

2. SHAP VALUES:
   - Individual: Each patient gets a SHAP value showing how AI2RhADa affected
     THEIR specific prediction
   - Directional: Positive SHAP = increases probability, Negative = decreases
   - Class-specific: Separate SHAP values for each ASIA grade

SHAP IMPACT BY CLASS (Mean absolute SHAP values):
"""

for grade_num in sorted(y.unique()):
    class_idx = list(model.classes_).index(grade_num)
    avg_abs = np.abs(shap_values[class_idx][:, ai2rhada_idx]).mean()
    avg_signed = shap_values[class_idx][:, ai2rhada_idx].mean()
    report += f"\nGrade {grade_num} ({ASIA_GRADE_MAP[grade_num]}):\n"
    report += f"  - Average impact magnitude: {avg_abs:.4f}\n"
    report += f"  - Average impact direction: {avg_signed:+.4f}\n"
    report += f"  - Correlation with AI2RhADa value: {correlations[grade_num]:+.4f}\n"

report += """

INTERPRETATION OF SHAP IMPACT:
"""

for grade_num in sorted(y.unique()):
    class_idx = list(model.classes_).index(grade_num)
    avg_signed = shap_values[class_idx][:, ai2rhada_idx].mean()
    corr = correlations[grade_num]
    
    if avg_signed > 0:
        direction = "INCREASES"
    else:
        direction = "DECREASES"
    
    if corr > 0:
        trend = "Higher AI2RhADa values → stronger effect"
    elif corr < 0:
        trend = "Lower AI2RhADa values → stronger effect"
    else:
        trend = "No clear linear relationship"
    
    report += f"\nGrade {grade_num} ({ASIA_GRADE_MAP[grade_num]}):\n"
    report += f"  - AI2RhADa {direction} the probability of this grade\n"
    report += f"  - {trend}\n"

report += f"""

{'='*80}
KEY TAKEAWAYS
{'='*80}

1. ACTUAL TREND IN DATA:
   See the descriptive statistics above for mean days by grade

2. FEATURE IMPORTANCE (7.5%):
   - AI2RhADa is moderately important overall
   - Ranks among top features but not #1
   - Useful across multiple classes

3. SHAP IMPACT:
   - Shows CLASS-SPECIFIC and DIRECTIONAL effects
   - Some grades are more sensitive to AI2RhADa than others
   - The relationship can be non-linear and complex

4. PRACTICAL IMPLICATION:
   Time to rehabilitation admission matters for predicting outcomes,
   but its effect varies by injury severity and interacts with other factors.

{'='*80}
VISUALIZATIONS GENERATED:
{'='*80}

1. AI2RhADa_distributions_by_grade.png
   → Shows how AI2RhADa is distributed for each ASIA grade

2. AI2RhADa_boxplot_by_grade.png
   → Compares AI2RhADa across grades (shows median, IQR, outliers)

3. AI2RhADa_mean_by_grade.png
   → Shows mean AI2RhADa for each grade with error bars

4. AI2RhADa_shap_dependency_by_grade.png
   → Shows how the model uses AI2RhADa to predict each grade
   → Each panel shows the relationship for one specific grade

5. AI2RhADa_shap_summary_by_grade.png
   → Panel A: How much does AI2RhADa matter for each grade?
   → Panel B: Does it increase or decrease probability for each grade?

{'='*80}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

# Save the report
with open('AI2RhADa_analysis_report.txt', 'w') as f:
    f.write(report)

print(report)
print("\n✓ Report saved to 'AI2RhADa_analysis_report.txt'")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. AI2RhADa_distributions_by_grade.png")
print("  2. AI2RhADa_boxplot_by_grade.png")
print("  3. AI2RhADa_mean_by_grade.png")
print("  4. AI2RhADa_shap_dependency_by_grade.png")
print("  5. AI2RhADa_shap_summary_by_grade.png")
print("  6. AI2RhADa_analysis_report.txt")
print("\nOpen these files to see the detailed trends and relationships!")

