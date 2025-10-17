"""
Create UPDATED comprehensive PDF report with CLEAN models (no data leakage)
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
print("CREATING UPDATED PDF WITH CLEAN MODELS (NO DATA LEAKAGE)")
print("="*70)

# Load summaries
print("\nLoading statistical outputs...")

with open('motor_clean_model_summary.txt', 'r') as f:
    motor_summary = f.read()

with open('impairment_model_summary.txt', 'r') as f:
    impairment_summary = f.read()

with open('auc_statistics.json', 'r') as f:
    auc_stats = json.load(f)

print("✓ Statistical outputs loaded")

# Define updated figure structure
figures = {
    'Model 1: ASIA Motor Score Prediction (Clean - No Data Leakage)': [
        {
            'file': 'motor_clean_predictions.png',
            'caption': 'Figure 1: Actual vs. Predicted ASIA Motor Scores at Discharge (Clean Model). (Left) Scatter plot showing good agreement between actual and predicted scores using ONLY admission features (R² = 0.812). The red dashed line represents perfect prediction. (Right) Residual plot showing randomly distributed errors with no systematic bias. This model uses NO discharge features, making it truly predictive.',
            'key_stats': 'R² = 0.8122, RMSE = 11.69, MAE = 7.64 (Admission features only)'
        },
        {
            'file': 'motor_clean_distributions.png',
            'caption': 'Figure 2: Distribution Analysis of Clean Motor Score Predictions. (Left) Comparison of actual vs. predicted score distributions showing similar patterns. (Right) Prediction error distribution is approximately normally distributed with mean near zero, indicating unbiased predictions. All predictions based on admission-time features.',
            'key_stats': 'Mean Error ≈ 0, Std Error = 11.69, No data leakage'
        },
        {
            'file': 'motor_clean_feature_importance.png',
            'caption': 'Figure 3: Top 20 Feature Importance for Clean Motor Score Prediction. Features ranked by their contribution using ONLY admission/injury-time data. AASATotA (admission score, 42.9%) and AASAImAd (admission impairment, 19.7%) are most important. NO discharge features used, ensuring true predictive value.',
            'key_stats': 'Top 3: AASATotA (42.9%), AASAImAd (19.7%), AFScorRb (8.5%)'
        },
        {
            'file': 'shap_summary_motor_clean.png',
            'caption': 'Figure 4: SHAP Summary Plot for Clean Motor Score Prediction. Each point represents a patient, colored by feature value (red=high, blue=low). Features ordered by importance. Uses ONLY admission features. Shows admission total score (AASATotA) has strongest positive impact. High admission scores lead to high predicted discharge scores.',
            'key_stats': 'SHAP analysis on 1,000 patients, admission features only'
        },
        {
            'file': 'shap_bar_motor_clean.png',
            'caption': 'Figure 5: SHAP Feature Importance (Bar Plot) for Clean Motor Score Model. Mean absolute SHAP values show average impact on predictions. Confirms admission total score and admission impairment are dominant predictors. All features are from admission time - no data leakage.',
            'key_stats': 'Based on SHAP values, truly predictive features'
        }
    ],
    'Model 2: ASIA Impairment Grade Classification': [
        {
            'file': 'impairment_confusion_matrix.png',
            'caption': 'Figure 6: Confusion Matrix for ASIA Impairment Grade Classification. Normalized heatmap showing classification accuracy. Grade E (normal function) achieves 98% recall. Grade D shows lower accuracy due to severe class imbalance (n=105, 0.7%). Uses only admission features.',
            'key_stats': 'Overall Accuracy = 82.6%, Weighted F1 = 82.3%'
        },
        {
            'file': 'impairment_class_distributions.png',
            'caption': 'Figure 7: Class Distribution Comparison for Impairment Classification. (Left) Actual distribution in test set. (Right) Predicted distribution. Model successfully captures the class imbalance pattern, with Grade E (48.7%) and Grade C (30.3%) as dominant categories.',
            'key_stats': 'Class balance maintained in predictions'
        },
        {
            'file': 'impairment_per_class_performance.png',
            'caption': 'Figure 8: Per-Class Performance Metrics for Impairment Classification. Precision, recall, and F1-scores for each ASIA grade. Grade E shows highest performance (F1=0.94) due to larger sample size. Grade D shows lower performance (F1=0.39) due to severe class imbalance.',
            'key_stats': 'Weighted Precision = 82.7%, Weighted Recall = 82.6%'
        },
        {
            'file': 'roc_curves_enhanced_model2.png',
            'caption': 'Figure 9: ROC Curves for Multi-Class Impairment Classification. Individual curves for each ASIA grade plus micro-average performance. All grades achieve AUC > 0.85, with Grade E reaching 0.99. Micro-average AUC = 0.960 indicates excellent discrimination ability. Uses only admission features.',
            'key_stats': f"Micro-AUC = {auc_stats['Model 2 - Impairment Classification']['Micro-Average AUC']:.3f}, Weighted-AUC = {auc_stats['Model 2 - Impairment Classification']['Weighted-Average AUC']:.3f}"
        },
        {
            'file': 'impairment_feature_importance.png',
            'caption': 'Figure 10: Top 20 Feature Importance for Impairment Classification. AASAImAd (admission impairment) dominates with 38.3% importance, indicating initial injury severity is the strongest predictor of discharge impairment. Time to rehabilitation (14.2%) is second most important.',
            'key_stats': 'Top 3: AASAImAd (38.3%), AI2RhADa (14.2%), AUMVAdm (7.5%)'
        },
        {
            'file': 'shap_summary_model2_impairment.png',
            'caption': 'Figure 11: SHAP Summary Plot for Impairment Classification. Multi-class SHAP values showing feature impact across all ASIA grades. Admission impairment (AASAImAd) shows strongest influence. Red indicates higher feature values, blue indicates lower values. All features from admission time.',
            'key_stats': 'SHAP analysis on 1,000 patients, admission features only'
        },
        {
            'file': 'shap_bar_model2_impairment.png',
            'caption': 'Figure 12: SHAP Feature Importance (Bar Plot) for Impairment Classification. Mean absolute SHAP values averaged across all classes. Confirms admission impairment as dominant predictor. Demographic factors show minimal importance compared to clinical measures.',
            'key_stats': 'Mean |SHAP| values across all predictions'
        }
    ]
}

# Create PDF
pdf_filename = 'ML_Models_UPDATED_Research_Report.pdf'
print(f"\nCreating PDF: {pdf_filename}")

with PdfPages(pdf_filename) as pdf:
    
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    print("  Creating title page...")
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title_text = """
    
    
    Machine Learning Models for Predicting
    Outcomes in Traumatic Spinal Cord Injury
    
    UPDATED: Clean Models (No Data Leakage)
    
    Comprehensive Figures and Statistical Analysis
    
    
    
    Two Random Forest Models (Both Truly Predictive):
    
    Model 1: ASIA Motor Score Prediction (Regression)
    • R² = 0.8122, RMSE = 11.69
    • 10,543 patients, 26 admission features
    • ✓ NO discharge features - truly predictive
    
    Model 2: ASIA Impairment Grade Classification
    • Accuracy = 82.6%, AUC = 94.2%
    • 15,053 patients, 26 admission features
    • ✓ NO discharge features - truly predictive
    
    
    
    """
    
    footer_text = f"""
    Generated: {datetime.now().strftime('%B %d, %Y')}
    
    UPDATED VERSION - Data Leakage Corrected
    Both models use ONLY admission/injury-time features
    Framework: scikit-learn Random Forest
    Feature Importance: SHAP (SHapley Additive exPlanations)
    """
    
    ax.text(0.5, 0.70, title_text, transform=ax.transAxes,
            fontsize=13, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
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
    UPDATED: Clean Models (No Data Leakage)
    
    Model 1: ASIA Motor Score Prediction - CLEAN
    
      Figure 1:  Actual vs. Predicted (Admission Features Only)
      Figure 2:  Distribution Analysis
      Figure 3:  Feature Importance (Admission Only)
      Figure 4:  SHAP Summary Plot (Beeswarm)
      Figure 5:  SHAP Feature Importance (Bar Plot)
      
    Model 2: ASIA Impairment Grade Classification
    
      Figure 6:  Confusion Matrix
      Figure 7:  Class Distribution Comparison
      Figure 8:  Per-Class Performance Metrics
      Figure 9:  ROC Curves (Multi-Class)
      Figure 10: Feature Importance
      Figure 11: SHAP Summary Plot (Beeswarm)
      Figure 12: SHAP Feature Importance (Bar Plot)
      
    Statistical Summaries & Analysis
    
      Model 1 Statistical Summary (Clean Model)
      Model 2 Statistical Summary
      Comparative Analysis (Both Models Now Clean)
      Clinical Implications & Recommendations
      SHAP Interpretation Guide
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
        
        color = 'lightgreen' if 'Clean' in section_title or 'Motor' in section_title else 'lightblue'
        ax.text(0.5, 0.5, section_title, transform=ax.transAxes,
                fontsize=16, va='center', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Individual figures
        for fig_info in section_figures:
            fig_file = fig_info['file']
            fig_caption = fig_info['caption']
            fig_stats = fig_info['key_stats']
            
            try:
                print(f"    Adding: {fig_file}")
                
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
    # MODEL 1 STATISTICAL SUMMARY (CLEAN)
    # ========================================================================
    print("\n  Creating Model 1 statistical summary (clean)...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Model 1: Statistical Summary (CLEAN - No Data Leakage)', 
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    stats_text = """
MODEL 1: ASIA MOTOR SCORE PREDICTION (CLEAN)

✓ UPDATED: NO DATA LEAKAGE - TRULY PREDICTIVE

Dataset Information:
  • Total Patients: 10,543
  • Features: 26 (ADMISSION/INJURY-TIME ONLY)
  • Target: AASATotD (0-100 scale)
  • Training/Test Split: 80/20

KEY DIFFERENCE FROM PREVIOUS VERSION:
  ✓ Uses ONLY admission and injury-time features
  ✓ NO discharge features included
  ✓ Truly predictive - can be used at admission
  ✓ Lower R² is EXPECTED and REALISTIC

Performance Metrics:
  Test Set Performance:
    R² Score:        0.8122  (explains 81.2% of variance) ✓
    RMSE:           11.69    (root mean squared error)
    MAE:             7.64    (mean absolute error)
    
  Cross-Validation (5-fold):
    Mean R²:         0.8149 ± 0.0335
    
Comparison to Old Model with Data Leakage:
  Old Model (with discharge features):
    R² = 0.905 (but NOT truly predictive)
  New Model (admission only):
    R² = 0.812 (TRULY predictive) ✓✓✓
    
Top 5 Predictive Features (Admission Only):
  1. AASATotA  (42.9%) - ASIA total at admission ✓
  2. AASAImAd  (19.7%) - ASIA impairment at admission ✓
  3. AFScorRb   (8.5%) - Functional score (rehab baseline) ✓
  4. ANurLvlA   (6.7%) - Neurological level at admission ✓
  5. AUMVAdm    (5.4%) - Upper motor vehicle at admission ✓

Model Characteristics:
  • Algorithm: Random Forest Regressor (200 trees)
  • Max Depth: 20
  • Features: 26 admission-time features

Clinical Utility:
  ✓ Can predict discharge motor scores AT ADMISSION
  ✓ Useful for early patient counseling
  ✓ Helps set realistic expectations
  ✓ Guides treatment planning and resource allocation
  ✓ True predictive value for clinical decision-making

Performance Context:
  • R² of 0.812 means model explains 81.2% of variance
  • Average prediction error is ±11.7 points (on 0-100 scale)
  • This is EXCELLENT for admission-only prediction
  • Previous R² of 0.905 was inflated by data leakage
    """
    
    ax.text(0.05, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace')
    
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

✓ CLEAN MODEL - ADMISSION FEATURES ONLY

Dataset Information:
  • Total Patients: 15,053
  • Features: 26 (admission-time features only)
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
    Accuracy:        82.60%  ✓
    F1 (Weighted):   82.34%
    F1 (Macro):      65.89%
    Precision:       82.68%
    Recall:          82.60%
    AUC (Weighted):  94.17%  ★★★★★
    AUC (Micro):     96.04%
    
  Cross-Validation (5-fold):
    Mean Accuracy:   83.44% ± 0.69%
    
Top 5 Predictive Features (Admission Only):
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

Clinical Utility:
  ✓ Predicts discharge impairment grade AT ADMISSION
  ✓ 82.6% accuracy for early outcome prediction
  ✓ Guides treatment intensity and resource needs
  ✓ Helps set realistic patient expectations
  ✓ Stratifies patients for clinical trials
    """
    
    ax.text(0.05, 0.95, stats_text2, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # COMPARATIVE ANALYSIS (UPDATED)
    # ========================================================================
    print("  Creating comparative analysis...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Comparative Analysis & Clinical Implications (UPDATED)', 
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    comparison_text = """
COMPARATIVE ANALYSIS - BOTH MODELS NOW CLEAN

✓ UPDATED: Both models now use ONLY admission features
✓ NO data leakage in either model
✓ Both models are TRULY PREDICTIVE

Performance Comparison:
  Model 1 (Motor Score - Clean):
    • Good accuracy (R² = 0.812) ✓
    • Realistic prediction error (RMSE = 11.7 points)
    • Truly predictive (admission features only) ✓
    • Average error ±11.7 points on 0-100 scale
    
  Model 2 (Impairment Grade - Clean):
    • Very good accuracy (82.6%) ✓
    • Excellent discrimination (AUC = 94.2%) ★★★★★
    • Truly predictive (admission features only) ✓
    • Strong per-class performance (except Grade D)

Feature Importance Insights:
  Model 1 (Motor Score):
    • Admission motor score (AASATotA) dominates (42.9%)
    • Admission impairment (AASAImAd) second (19.7%)
    • ALL features from admission time ✓
    
  Model 2 (Impairment Grade):
    • Admission impairment (AASAImAd) key (38.3%)
    • Time to rehab matters significantly (14.2%)
    • ALL features from admission time ✓

CLINICAL IMPLICATIONS

For Early Prediction (at Admission):
  ✓ BOTH models can now be used for early counseling
  ✓ Model 1: Predicts continuous motor score (0-100)
  ✓ Model 2: Predicts categorical ASIA grade (A-E)
  
  Choose based on clinical need:
  • Continuous outcome → Use Model 1 (R² = 0.812)
  • Categorical grade → Use Model 2 (Acc = 82.6%)
  • Both provide complementary information

Performance Context:
  Previous Model 1 (with data leakage):
    R² = 0.905, but NOT truly predictive ✗
    
  Current Model 1 (clean):
    R² = 0.812, TRULY predictive ✓✓✓
    
  The drop in R² from 0.905 to 0.812 is EXPECTED when
  removing discharge features. This represents the
  REALISTIC predictive power from admission data.

Key Findings:
  1. Admission severity is the strongest predictor
     (38-43% importance across both models)
     
  2. Early rehabilitation matters - time to rehab
     significantly impacts outcomes (14.2% importance)
     
  3. Motor scores are highly predictable from admission
     data (R² = 0.812, 81% variance explained)
     
  4. Impairment grades are well-classified from
     admission data (82.6% accuracy, 94.2% AUC)
     
  5. Demographic factors have minimal impact compared
     to clinical measures at admission

Recommendations for Clinical Use:
  ✓ Use BOTH models at admission for comprehensive
    outcome prediction
  ✓ Model 1 for continuous motor score prediction
  ✓ Model 2 for categorical impairment classification
  ✓ Set realistic patient expectations early
  ✓ Expedite rehabilitation admission (time matters!)
  ✓ Focus resources on patients with severe admission
    impairment (strongest predictor)

Model Selection Guide:
  Research Question: What predicts motor recovery?
    → Use Model 1 (continuous, R² = 0.812)
    
  Clinical Decision: What impairment grade at discharge?
    → Use Model 2 (categorical, 82.6% accuracy)
    
  Patient Counseling: What to expect at discharge?
    → Use BOTH models for comprehensive picture

Quality Assurance:
  ✓ Both models validated with cross-validation
  ✓ No data leakage in either model
  ✓ Admission-only features ensure true prediction
  ✓ Performance realistic and clinically useful
  ✓ SHAP analysis confirms feature interpretability
    """
    
    ax.text(0.03, 0.98, comparison_text, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # ROC AUC DETAILS (same as before)
    # ========================================================================
    print("  Creating ROC AUC details page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('ROC Curve Analysis - Model 2', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    per_class_auc = auc_stats['Model 2 - Impairment Classification']['Per-Class AUC']
    auc_text = f"""
RECEIVER OPERATING CHARACTERISTIC (ROC) CURVES
Model 2: ASIA Impairment Grade Classification

Individual Class AUC Scores:
"""
    for grade, auc_val in per_class_auc.items():
        stars = '★' * int(auc_val * 5)
        auc_text += f"  {grade}: {auc_val:.4f}  {stars}\n"
    
    auc_text += f"""
Aggregate AUC Scores:
  Micro-Average:    {auc_stats['Model 2 - Impairment Classification']['Micro-Average AUC']:.4f}  ★★★★★
  Macro-Average:    {auc_stats['Model 2 - Impairment Classification']['Macro-Average AUC']:.4f}  ★★★★★
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
  • Grade E (normal) has near-perfect discrimination (0.99)
  • Grade D has good AUC (0.89) despite small sample
  • Micro-average AUC of 0.960 indicates excellent
    classification ability across all grades
  • Model reliably distinguishes between grades
    """
    
    ax.text(0.08, 0.95, auc_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # SHAP INTERPRETATION (updated)
    # ========================================================================
    print("  Creating SHAP interpretation page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('SHAP Analysis Interpretation (Updated Models)', 
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    shap_text = """
SHAP (SHapley Additive exPlanations) ANALYSIS
Applied to Clean Models (Admission Features Only)

What is SHAP?
  SHAP assigns each feature an importance value for predictions,
  showing how much that feature contributed to pushing the
  prediction higher or lower. Based on game theory (Shapley values).

SHAP Summary Plot (Beeswarm):
  • Each dot = one patient
  • Y-axis = features (ordered by importance)
  • X-axis = SHAP value (impact on prediction)
  • Color = feature value (red = high, blue = low)
  
  Interpretation:
    - Right (positive SHAP) → increases prediction
    - Left (negative SHAP) → decreases prediction
    - Color shows whether high/low values cause effect
    - Width shows distribution across patients

Key Insights from Clean Models:
  
  Model 1 (Motor Score - Clean):
    • High admission motor scores → high discharge scores
    • Admission impairment grade strongly influences outcome
    • Functional score at rehab baseline matters
    • ALL features from admission - truly predictive ✓
    
  Model 2 (Impairment Grade):
    • Admission impairment dominates predictions (38%)
    • Longer time to rehab → worse outcomes
    • Age has non-linear effects
    • Neurological level shows clear importance
    • ALL features from admission - truly predictive ✓

Comparison to Previous Model with Data Leakage:
  Old Model 1:
    • Discharge features dominated SHAP plots
    • Not truly predictive ✗
    
  New Model 1 (Clean):
    • Admission features dominate SHAP plots
    • Truly predictive ✓✓✓

Clinical Applications of SHAP:
  1. Individual patient explanations
     → "Your admission score of X predicts..."
     
  2. Feature importance validation
     → Confirms admission severity is key
     
  3. Non-linear relationship discovery
     → Age effects are not linear
     
  4. Feature interaction detection
     → Combinations of factors matter
     
  5. Model trust and interpretability
     → Shows how predictions are made

References:
  Lundberg, S. M., & Lee, S. I. (2017). A unified approach
  to interpreting model predictions. NeurIPS.
    """
    
    ax.text(0.05, 0.98, shap_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # FINAL SUMMARY PAGE
    # ========================================================================
    print("  Creating final summary page...")
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle('Final Summary: Clean Models for Clinical Use', 
                 fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    final_text = """
FINAL SUMMARY: CLEAN PREDICTIVE MODELS

UPDATED REPORT - DATA LEAKAGE CORRECTED

What Changed:
  ✗ Previous Model 1 used discharge features (data leakage)
  ✓ Updated Model 1 uses ONLY admission features
  ✓ Model 2 was already clean (no changes needed)
  ✓ Both models now truly predictive

Final Model Performance:

  Model 1: ASIA Motor Score (Clean)
    R² Score:        0.8122  (81.2% variance explained)
    RMSE:           11.69 points
    MAE:             7.64 points
    Features:        26 admission-time features
    
    Clinical Use:
    ✓ Predict motor score at discharge
    ✓ Continuous outcome (0-100 scale)
    ✓ Use at admission for counseling
    ✓ Average error ±11.7 points
    
  Model 2: ASIA Impairment Grade (Clean)
    Accuracy:       82.60%
    AUC:            94.17%  ★★★★★
    F1 (Weighted):  82.34%
    Features:       26 admission-time features
    
    Clinical Use:
    ✓ Predict ASIA grade at discharge
    ✓ Categorical outcome (A, B, C, D, E)
    ✓ Use at admission for counseling
    ✓ Excellent discrimination (AUC = 0.942)

Key Clinical Findings:
  1. Admission severity predicts discharge outcomes
     (38-43% feature importance)
     
  2. Time to rehabilitation matters significantly
     (14% importance in Model 2)
     
  3. Both continuous and categorical predictions
     available for comprehensive assessment
     
  4. Models are realistic and clinically useful
     (no inflated performance from data leakage)
     
  5. Early intervention and accurate admission
     assessment are critical

Recommendations:
  ✓ Use Model 1 for motor score prediction (R² = 0.812)
  ✓ Use Model 2 for impairment classification (Acc = 82.6%)
  ✓ Apply at admission for early counseling
  ✓ Set realistic expectations with patients
  ✓ Expedite rehabilitation admission
  ✓ Focus resources on severe admission cases

Quality Assurance Checklist:
  ✓ Both models use admission features only
  ✓ No data leakage in either model
  ✓ Cross-validated performance
  ✓ SHAP analysis for interpretability
  ✓ ROC curves for discrimination assessment
  ✓ Realistic and clinically useful predictions
  ✓ Ready for prospective validation

Next Steps for Implementation:
  1. Prospective validation on new patients
  2. Integration into clinical workflow
  3. User interface development
  4. Ongoing model monitoring and updating
  5. Publication of results

This report provides publication-ready figures and
comprehensive statistical analysis for both models.
All models are clean, validated, and ready for
clinical research or implementation.
    """
    
    ax.text(0.05, 0.98, final_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # Save metadata
    d = pdf.infodict()
    d['Title'] = 'ML Models for SCI - UPDATED (No Data Leakage)'
    d['Author'] = 'Random Forest Analysis - Clean Models'
    d['Subject'] = 'Research Figures - Corrected for Data Leakage'
    d['Keywords'] = 'Machine Learning, Spinal Cord Injury, ASIA, Random Forest, SHAP, Clean Models'
    d['CreationDate'] = datetime.now()

print(f"\n✓ UPDATED PDF created successfully: {pdf_filename}")
print(f"\nTotal pages: ~22")
print("\nPDF Contents:")
print("  • Title page (UPDATED)")
print("  • Table of contents")
print("  • 12 figures with detailed captions")
print("  • Model 1: CLEAN motor score model (no data leakage)")
print("  • Model 2: Impairment classification (already clean)")
print("  • Updated statistical summaries")
print("  • Updated comparative analysis")
print("  • ROC AUC analysis")
print("  • SHAP interpretation")
print("  • Final summary page")
print("\n" + "="*70)
print("✓ UPDATED COMPREHENSIVE PDF COMPLETE!")
print("="*70)
print(f"\nFile location: {pdf_filename}")
print("\n✓ Both models now clean and truly predictive!")
print("✓ No data leakage in either model!")
print("✓ Ready for research publication!")

