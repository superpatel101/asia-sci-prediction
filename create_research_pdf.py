"""
Create a comprehensive PDF report for research publication with all figures,
captions, statistical outputs, and insights.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from PIL import Image
import pandas as pd
import json
import numpy as np
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

print("="*70)
print("CREATING COMPREHENSIVE PDF REPORT FOR RESEARCH PUBLICATION")
print("="*70)

# Load statistical outputs
print("\nLoading statistical outputs...")

# Load Model 1 summary
with open('model_summary_report.txt', 'r') as f:
    model1_summary = f.read()

# Load Model 2 summary
with open('impairment_model_summary.txt', 'r') as f:
    model2_summary = f.read()

# Load AUC statistics
with open('auc_statistics.json', 'r') as f:
    auc_stats = json.load(f)

print("✓ Statistical outputs loaded")

# Define figure structure
figures = {
    'Model 1: Motor Score Prediction (Regression)': [
        {
            'file': 'model_predictions.png',
            'caption': 'Figure 1: Actual vs. Predicted ASIA Motor Scores at Discharge. (Left) Scatter plot showing excellent agreement between actual and predicted motor scores (R² = 0.905). The red dashed line represents perfect prediction. (Right) Residual plot showing randomly distributed errors with no systematic bias.',
            'key_stats': 'R² = 0.9053, RMSE = 8.30, MAE = 5.42'
        },
        {
            'file': 'distributions.png',
            'caption': 'Figure 2: Distribution Analysis of Motor Score Predictions. (Left) Comparison of actual vs. predicted score distributions showing similar patterns. (Right) Prediction error distribution is approximately normally distributed with mean near zero, indicating unbiased predictions.',
            'key_stats': 'Mean Error ≈ 0, Std Error = 8.30'
        },
        {
            'file': 'feature_importance.png',
            'caption': 'Figure 3: Top 20 Feature Importance for Motor Score Prediction. Features are ranked by their contribution to model predictions. AASATotA (admission score) and discharge-time measures dominate, suggesting strong correlation but potential data leakage.',
            'key_stats': 'Top 3: AASATotA (26.7%), AASAImDs (17.3%), ABdMMDis (12.3%)'
        },
        {
            'file': 'shap_summary_model1_motor.png',
            'caption': 'Figure 4: SHAP Summary Plot for Motor Score Prediction. Each point represents a patient, colored by feature value (red=high, blue=low). Features are ordered by importance. Positive SHAP values increase predicted scores. Shows complex non-linear relationships between features and outcomes.',
            'key_stats': 'SHAP analysis on 1,000 randomly sampled patients'
        },
        {
            'file': 'shap_bar_model1_motor.png',
            'caption': 'Figure 5: SHAP Feature Importance (Bar Plot) for Motor Score Prediction. Mean absolute SHAP values indicate average impact on predictions. Complements traditional feature importance by accounting for feature interactions and directionality.',
            'key_stats': 'Based on SHAP values across all predictions'
        }
    ],
    'Model 2: Impairment Grade Classification': [
        {
            'file': 'impairment_confusion_matrix.png',
            'caption': 'Figure 6: Confusion Matrix for ASIA Impairment Grade Classification. Heatmap shows normalized classification accuracy. Diagonal elements represent correct predictions. Grade E (normal function) achieves 98% recall. Grade D shows lower accuracy due to class imbalance (n=105).',
            'key_stats': 'Overall Accuracy = 82.6%, Weighted F1 = 82.3%'
        },
        {
            'file': 'impairment_class_distributions.png',
            'caption': 'Figure 7: Class Distribution Comparison. (Left) Actual distribution in test set. (Right) Predicted distribution. Model successfully captures the class imbalance pattern, with Grade E (48.7%) and Grade C (30.3%) as dominant categories.',
            'key_stats': 'Class balance maintained in predictions'
        },
        {
            'file': 'impairment_per_class_performance.png',
            'caption': 'Figure 8: Per-Class Performance Metrics. Precision, recall, and F1-scores for each ASIA grade. Grade E shows highest performance (F1=0.94) due to larger sample size. Grade D shows lower performance (F1=0.39) due to severe class imbalance.',
            'key_stats': 'Weighted Precision = 82.7%, Weighted Recall = 82.6%'
        },
        {
            'file': 'roc_curves_enhanced_model2.png',
            'caption': 'Figure 9: ROC Curves for Multi-Class Impairment Classification. Individual curves for each ASIA grade plus micro-average performance. All grades achieve AUC > 0.85, with Grade E reaching 0.99. Micro-average AUC = 0.960 indicates excellent discrimination ability.',
            'key_stats': f"Micro-AUC = {auc_stats['Model 2 - Impairment Classification']['Micro-Average AUC']:.3f}, Weighted-AUC = {auc_stats['Model 2 - Impairment Classification']['Weighted-Average AUC']:.3f}"
        },
        {
            'file': 'impairment_feature_importance.png',
            'caption': 'Figure 10: Top 20 Feature Importance for Impairment Classification. AASAImAd (admission impairment) dominates with 38.3% importance, indicating initial injury severity is the strongest predictor of discharge impairment. Time to rehabilitation (14.2%) is second most important.',
            'key_stats': 'Top 3: AASAImAd (38.3%), AI2RhADa (14.2%), AUMVAdm (7.5%)'
        },
        {
            'file': 'shap_summary_model2_impairment.png',
            'caption': 'Figure 11: SHAP Summary Plot for Impairment Classification. Multi-class SHAP values showing feature impact across all ASIA grades. Admission impairment (AASAImAd) shows strongest influence. Red indicates higher feature values, blue indicates lower values.',
            'key_stats': 'SHAP analysis on 1,000 randomly sampled patients'
        },
        {
            'file': 'shap_bar_model2_impairment.png',
            'caption': 'Figure 12: SHAP Feature Importance (Bar Plot) for Impairment Classification. Mean absolute SHAP values averaged across all classes. Confirms admission impairment as dominant predictor. Demographic factors show minimal importance compared to clinical measures.',
            'key_stats': 'Mean |SHAP| values across all predictions'
        }
    ]
}

# Create PDF
pdf_filename = 'ML_Models_Research_Figures_Report.pdf'
print(f"\nCreating PDF: {pdf_filename}")

with PdfPages(pdf_filename) as pdf:
    
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    print("  Creating title page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title_text = """
    
    
    Machine Learning Models for Predicting
    Outcomes in Traumatic Spinal Cord Injury
    
    Comprehensive Figures and Statistical Analysis
    
    
    
    Two Random Forest Models:
    
    Model 1: ASIA Motor Score Prediction (Regression)
    • R² = 0.9053, RMSE = 8.30
    • 10,543 patients, 32 features
    
    Model 2: ASIA Impairment Grade Classification
    • Accuracy = 82.6%, AUC = 94.2%
    • 15,053 patients, 26 features
    
    
    
    """
    
    footer_text = f"""
    Generated: {datetime.now().strftime('%B %d, %Y')}
    
    Framework: scikit-learn Random Forest
    Feature Importance: SHAP (SHapley Additive exPlanations)
    """
    
    ax.text(0.5, 0.70, title_text, transform=ax.transAxes,
            fontsize=14, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.text(0.5, 0.15, footer_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='center', style='italic')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # TABLE OF CONTENTS
    # ========================================================================
    print("  Creating table of contents...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Table of Contents', fontsize=16, fontweight='bold', y=0.98)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    toc_text = """
    Model 1: ASIA Motor Score Prediction (Regression)
    
      Figure 1:  Actual vs. Predicted Motor Scores
      Figure 2:  Distribution Analysis
      Figure 3:  Feature Importance (Traditional)
      Figure 4:  SHAP Summary Plot (Beeswarm)
      Figure 5:  SHAP Feature Importance (Bar Plot)
      
    Model 2: ASIA Impairment Grade Classification
    
      Figure 6:  Confusion Matrix
      Figure 7:  Class Distribution Comparison
      Figure 8:  Per-Class Performance Metrics
      Figure 9:  ROC Curves (Multi-Class)
      Figure 10: Feature Importance (Traditional)
      Figure 11: SHAP Summary Plot (Beeswarm)
      Figure 12: SHAP Feature Importance (Bar Plot)
      
    Statistical Summaries
    
      Model 1 Statistical Summary
      Model 2 Statistical Summary
      Comparative Analysis
      Clinical Implications
    """
    
    ax.text(0.1, 0.85, toc_text, transform=ax.transAxes,
            fontsize=11, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # FIGURES WITH CAPTIONS
    # ========================================================================
    
    for section_title, section_figures in figures.items():
        print(f"\n  Processing: {section_title}")
        
        # Section title page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.5, section_title, transform=ax.transAxes,
                fontsize=18, va='center', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Individual figures
        for fig_info in section_figures:
            fig_file = fig_info['file']
            fig_caption = fig_info['caption']
            fig_stats = fig_info['key_stats']
            
            try:
                print(f"    Adding: {fig_file}")
                
                # Create figure
                fig = plt.figure(figsize=(8.5, 11))
                gs = gridspec.GridSpec(20, 1, figure=fig, hspace=0.3)
                
                # Image
                ax_img = fig.add_subplot(gs[0:16, 0])
                img = Image.open(fig_file)
                ax_img.imshow(img)
                ax_img.axis('off')
                
                # Caption
                ax_caption = fig.add_subplot(gs[17:19, 0])
                ax_caption.axis('off')
                ax_caption.text(0.02, 0.95, fig_caption, transform=ax_caption.transAxes,
                              fontsize=10, va='top', ha='left', wrap=True)
                
                # Key statistics
                ax_stats = fig.add_subplot(gs[19:20, 0])
                ax_stats.axis('off')
                ax_stats.text(0.02, 0.8, f"Key Statistics: {fig_stats}", 
                            transform=ax_stats.transAxes,
                            fontsize=9, va='top', ha='left', style='italic',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"    ⚠ Warning: Could not load {fig_file}: {e}")
    
    # ========================================================================
    # MODEL 1 STATISTICAL SUMMARY
    # ========================================================================
    print("\n  Creating Model 1 statistical summary...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Model 1: Statistical Summary', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    stats_text = """
MODEL 1: ASIA MOTOR SCORE PREDICTION (REGRESSION)

Dataset Information:
  • Total Patients: 10,543
  • Features: 32 input features
  • Target: AASATotD (0-100 scale)
  • Training/Test Split: 80/20 (stratified)

Performance Metrics:
  Test Set Performance:
    R² Score:        0.9053  (explains 90.5% of variance)
    RMSE:            8.30    (root mean squared error)
    MAE:             5.42    (mean absolute error)
    
  Cross-Validation (5-fold):
    Mean R²:         0.9052 ± 0.0156
    
Top 5 Predictive Features:
  1. AASATotA  (26.7%) - ASIA total at admission
  2. AASAImDs  (17.3%) - ASIA impairment at discharge ⚠
  3. ABdMMDis  (12.3%) - Bowel/bladder at discharge ⚠
  4. AASAImAd   (9.8%) - ASIA impairment at admission
  5. AFScorDs   (8.0%) - Functional score at discharge ⚠

Model Characteristics:
  • Algorithm: Random Forest Regressor (200 trees)
  • Max Depth: 20
  • Class Weight: None (regression)

Important Note:
  ⚠ This model uses DISCHARGE features to predict discharge
     outcome, which may limit its predictive utility for 
     early patient counseling and treatment planning.
  
  ✓ Best used for: retrospective analysis, quality metrics,
     understanding feature relationships.
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # MODEL 2 STATISTICAL SUMMARY
    # ========================================================================
    print("  Creating Model 2 statistical summary...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Model 2: Statistical Summary', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    stats_text2 = """
MODEL 2: ASIA IMPAIRMENT GRADE CLASSIFICATION

Dataset Information:
  • Total Patients: 15,053
  • Features: 26 input features
  • Target: AASAImDs (Grades A, B, C, D, E)
  • Training/Test Split: 80/20 (stratified)
  • Class Distribution:
      Grade A (Complete):          10.35%
      Grade B (Incomplete-Sensory):  9.98%
      Grade C (Incomplete-Motor):   30.27%
      Grade D (Incomplete-Motor):    0.70% ⚠
      Grade E (Normal):             48.70%

Performance Metrics:
  Test Set Performance:
    Accuracy:        82.60%
    F1 (Weighted):   82.34%
    F1 (Macro):      65.89%
    Precision:       82.68%
    Recall:          82.60%
    AUC (Weighted):  94.17%  ★★★
    AUC (Micro):     96.04%
    
  Cross-Validation (5-fold):
    Mean Accuracy:   83.44% ± 0.69%
    
Top 5 Predictive Features:
  1. AASAImAd  (38.3%) - ASIA impairment at admission ✓
  2. AI2RhADa  (14.2%) - Days injury to rehab admission ✓
  3. AUMVAdm    (7.5%) - Upper motor vehicle at admission ✓
  4. ANurLvlA   (7.4%) - Neurological level at admission ✓
  5. AInjAge    (7.2%) - Age at injury ✓

Per-Class Performance (F1-Scores):
  Grade A: 0.66  |  Grade B: 0.49  |  Grade C: 0.81
  Grade D: 0.39  |  Grade E: 0.94

Model Characteristics:
  • Algorithm: Random Forest Classifier (200 trees)
  • Max Depth: 20
  • Class Weight: Balanced (handles imbalance)

Important Note:
  ✓ This model uses ADMISSION-TIME features only,
     making it truly predictive for early intervention.
  
  ✓ Best used for: early outcome prediction, treatment
     planning, patient counseling, resource allocation.
    """
    
    ax.text(0.05, 0.95, stats_text2, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================
    print("  Creating comparative analysis...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Comparative Analysis & Clinical Implications', 
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    comparison_text = """
COMPARATIVE ANALYSIS OF BOTH MODELS

Performance Comparison:
  Model 1 (Motor Score):
    • Excellent accuracy (R² = 0.905)
    • Low prediction error (RMSE = 8.3 points)
    • Limited predictive value (uses discharge data)
    
  Model 2 (Impairment Grade):
    • Very good accuracy (82.6%)
    • Excellent discrimination (AUC = 94.2%)
    • High predictive value (uses admission data only) ★

Feature Importance Insights:
  Model 1:
    • Discharge measures dominate (potential data leakage)
    • AASATotA (admission) still important (26.7%)
    
  Model 2:
    • Admission impairment grade is key (38.3%)
    • Time to rehab matters (14.2%)
    • Demographic factors less important (<3%)

CLINICAL IMPLICATIONS

For Early Prediction (at Admission):
  ✓ Use Model 2 (Impairment Classifier)
    - Provides ASIA grade predictions
    - 82.6% accuracy is clinically useful
    - Includes confidence estimates
    - Truly predictive (no data leakage)

For Retrospective Analysis:
  ✓ Use Model 1 (Motor Score)
    - Highly accurate (90.5% R²)
    - Continuous predictions
    - Good for quality metrics

Key Findings:
  1. Initial injury severity (admission impairment) is the
     strongest predictor of discharge outcomes (38.3%)
     
  2. Early rehabilitation matters - days from injury to
     rehab is second most important feature (14.2%)
     
  3. Motor score recovery is highly predictable when
     discharge information is available (R² = 0.905)
     
  4. Complete recovery (Grade E) is easiest to predict
     (F1 = 0.94, 98% recall)
     
  5. Demographic factors have minimal impact compared
     to clinical measures

Recommendations for Clinical Use:
  • Use Model 2 at admission for early counseling
  • Set realistic patient expectations based on admission grade
  • Expedite rehabilitation admission (time matters!)
  • Use Model 1 for retrospective quality assessment
  • Combine both for comprehensive outcome analysis

Limitations:
  • Model 1 has data leakage concerns
  • Model 2: Grade D predictions less reliable (n=105)
  • Dataset-specific; may need validation on new populations
  • Static predictions; don't model recovery trajectory
  • No confidence intervals (single point estimates)
    """
    
    ax.text(0.05, 0.98, comparison_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # ROC AUC DETAILS
    # ========================================================================
    print("  Creating ROC AUC details page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('ROC Curve Analysis - Model 2', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Format AUC statistics
    per_class_auc = auc_stats['Model 2 - Impairment Classification']['Per-Class AUC']
    auc_text = f"""
RECEIVER OPERATING CHARACTERISTIC (ROC) CURVES
Model 2: ASIA Impairment Grade Classification

Individual Class AUC Scores:
"""
    for grade, auc_val in per_class_auc.items():
        stars = '★' * int(auc_val * 5)  # Stars based on AUC
        auc_text += f"  {grade}: {auc_val:.4f}  {stars}\n"
    
    auc_text += f"""
Aggregate AUC Scores:
  Micro-Average:   {auc_stats['Model 2 - Impairment Classification']['Micro-Average AUC']:.4f}  ★★★★★
  Macro-Average:   {auc_stats['Model 2 - Impairment Classification']['Macro-Average AUC']:.4f}  ★★★★★
  Weighted-Average: {auc_stats['Model 2 - Impairment Classification']['Weighted-Average AUC']:.4f}  ★★★★★

Interpretation:
  AUC Score       Discrimination Ability
  0.90 - 1.00     Excellent  ★★★★★
  0.80 - 0.90     Good       ★★★★
  0.70 - 0.80     Fair       ★★★
  0.60 - 0.70     Poor       ★★
  0.50 - 0.60     Fail       ★

Clinical Significance:
  • All grades achieve AUC > 0.85 (good to excellent)
  • Grade E (normal function) has near-perfect discrimination (0.99)
  • Grade D has lower AUC (0.89) but still good performance
  • Micro-average AUC of 0.960 indicates excellent overall
    classification ability across all grades
  • Model can reliably distinguish between impairment grades
    
One-vs-Rest (OVR) Strategy:
  • Each grade treated as binary classification
  • Positive class = specific grade
  • Negative class = all other grades
  • AUC measures separability for each grade

Micro-Average:
  • Aggregates contributions of all classes
  • Weights classes by support (sample size)
  • Best for imbalanced datasets
  
Weighted-Average:
  • Averages per-class AUC weighted by support
  • Accounts for class imbalance
  • Recommended metric for reporting
    """
    
    ax.text(0.08, 0.95, auc_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # SHAP INTERPRETATION
    # ========================================================================
    print("  Creating SHAP interpretation page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('SHAP Analysis Interpretation', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    shap_text = """
SHAP (SHapley Additive exPlanations) ANALYSIS

What is SHAP?
  SHAP is a game-theoretic approach to explain machine learning
  model predictions. It assigns each feature an importance value
  (SHAP value) for a particular prediction, showing how much that
  feature contributed to pushing the prediction higher or lower.

SHAP Summary Plot (Beeswarm):
  • Each dot = one patient
  • Y-axis = features (ordered by importance)
  • X-axis = SHAP value (impact on prediction)
  • Color = feature value (red = high, blue = low)
  
  Interpretation:
    - Points to the right (positive SHAP) increase prediction
    - Points to the left (negative SHAP) decrease prediction
    - Color shows whether high/low feature values cause the effect
    - Width shows how many patients have that impact

SHAP Feature Importance (Bar):
  • Shows mean absolute SHAP value for each feature
  • Indicates average magnitude of impact
  • Complements traditional feature importance
  • Accounts for feature interactions

Key Advantages of SHAP:
  1. Model-agnostic (works with any ML model)
  2. Theoretically sound (based on Shapley values)
  3. Shows directionality (positive/negative impact)
  4. Captures feature interactions
  5. Provides local explanations (individual predictions)
  6. Globally consistent

Reading the Plots:
  Model 1 (Motor Score):
    • High admission scores → higher discharge scores
    • Discharge measures have strong direct effects
    • Complex interactions between clinical variables
    
  Model 2 (Impairment Grade):
    • Admission impairment dominates predictions
    • Longer time to rehab → worse outcomes
    • Age has non-linear effects
    • Neurological level shows importance

Clinical Insights from SHAP:
  • Admission severity is paramount (38% importance)
  • Early intervention timing matters significantly
  • Non-linear relationships exist (not captured by
    simple statistics)
  • Feature interactions are important
  • Patient-specific factors create prediction variability

References:
  Lundberg, S. M., & Lee, S. I. (2017). A unified approach
  to interpreting model predictions. NeurIPS.
    """
    
    ax.text(0.08, 0.98, shap_text, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Save metadata
    d = pdf.infodict()
    d['Title'] = 'ML Models for Spinal Cord Injury Outcome Prediction'
    d['Author'] = 'Random Forest Analysis'
    d['Subject'] = 'Research Figures and Statistical Analysis'
    d['Keywords'] = 'Machine Learning, Spinal Cord Injury, ASIA Score, Random Forest, SHAP'
    d['CreationDate'] = datetime.now()

print(f"\n✓ PDF created successfully: {pdf_filename}")
print(f"\nTotal pages: ~20")
print("\nPDF Contents:")
print("  • Title page")
print("  • Table of contents")
print("  • 12 figures with detailed captions")
print("  • Statistical summaries for both models")
print("  • Comparative analysis")
print("  • ROC AUC detailed analysis")
print("  • SHAP interpretation guide")
print("\n" + "="*70)
print("✓ COMPREHENSIVE PDF REPORT COMPLETE!")
print("="*70)
print(f"\nFile location: {pdf_filename}")
print("Ready for research publication!")

