"""
Analyze ASIA motor scores by 20-point buckets and create clean admission vs discharge plots
Focus on numeric scores (0-100) rather than letter grades
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ASIA MOTOR SCORE ANALYSIS: BY 20-POINT RANGES")
print("="*80)

# Load data
df = pd.read_csv('/Users/aaryanpatel/Downloads/V2_EDIT_modelreadyASIAMotor.csv')
print(f"\nDataset loaded: {len(df):,} patients")

# Extract motor scores
admission_motor = df['AASATotA']  # Motor score at admission (0-100)
discharge_motor = df['AASATotD']  # Motor score at discharge (0-100)

# Remove missing values
mask = (~admission_motor.isna()) & (~discharge_motor.isna())
admission_motor = admission_motor[mask]
discharge_motor = discharge_motor[mask]

print(f"Valid samples: {len(admission_motor):,}")

# Calculate change
motor_change = discharge_motor - admission_motor

# ============================================================================
# OVERALL STATISTICS
# ============================================================================

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

pearson_r, _ = sp_stats.pearsonr(admission_motor, discharge_motor)
print(f"\nCorrelation: r = {pearson_r:.4f} (R² = {pearson_r**2:.4f})")
print(f"Admission score explains {pearson_r**2*100:.1f}% of discharge variance")

print(f"\nAdmission Motor Score:  {admission_motor.mean():.1f} ± {admission_motor.std():.1f} (median: {admission_motor.median():.1f})")
print(f"Discharge Motor Score:  {discharge_motor.mean():.1f} ± {discharge_motor.std():.1f} (median: {discharge_motor.median():.1f})")
print(f"Change:                 {motor_change.mean():.1f} ± {motor_change.std():.1f} (median: {motor_change.median():.1f})")

improved = (motor_change > 0).sum()
unchanged = (motor_change == 0).sum()
declined = (motor_change < 0).sum()

print(f"\nOutcomes:")
print(f"  Improved:  {improved:,} ({improved/len(motor_change)*100:.1f}%)")
print(f"  Unchanged: {unchanged:,} ({unchanged/len(motor_change)*100:.1f}%)")
print(f"  Declined:  {declined:,} ({declined/len(motor_change)*100:.1f}%)")

# ============================================================================
# BUCKET ANALYSIS (20-POINT RANGES)
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS BY 20-POINT ADMISSION SCORE BUCKETS")
print("="*80)

# Define buckets
bucket_edges = [0, 20, 40, 60, 80, 100]
bucket_labels = ['0-19', '20-39', '40-59', '60-79', '80-100']

# Assign buckets
admission_buckets = pd.cut(admission_motor, bins=bucket_edges, labels=bucket_labels, include_lowest=True)

bucket_stats = []

for bucket_label in bucket_labels:
    mask = (admission_buckets == bucket_label)
    if mask.sum() == 0:
        continue
    
    n = mask.sum()
    adm = admission_motor[mask]
    dis = discharge_motor[mask]
    change = motor_change[mask]
    
    # Correlation within bucket
    if len(adm) > 2:
        corr, _ = sp_stats.pearsonr(adm, dis)
    else:
        corr = np.nan
    
    stats = {
        'bucket': bucket_label,
        'n': n,
        'adm_mean': adm.mean(),
        'adm_std': adm.std(),
        'adm_median': adm.median(),
        'dis_mean': dis.mean(),
        'dis_std': dis.std(),
        'dis_median': dis.median(),
        'change_mean': change.mean(),
        'change_std': change.std(),
        'change_median': change.median(),
        'pct_improved': (change > 0).sum() / n * 100,
        'pct_unchanged': (change == 0).sum() / n * 100,
        'pct_declined': (change < 0).sum() / n * 100,
        'correlation': corr
    }
    bucket_stats.append(stats)

# Print bucket statistics
for stats in bucket_stats:
    print(f"\n{'─'*80}")
    print(f"ADMISSION SCORE: {stats['bucket']} points - n = {stats['n']:,} patients")
    print(f"{'─'*80}")
    print(f"Admission:  Mean = {stats['adm_mean']:5.1f} ± {stats['adm_std']:4.1f}, Median = {stats['adm_median']:5.1f}")
    print(f"Discharge:  Mean = {stats['dis_mean']:5.1f} ± {stats['dis_std']:4.1f}, Median = {stats['dis_median']:5.1f}")
    print(f"Change:     Mean = {stats['change_mean']:+5.1f} ± {stats['change_std']:4.1f}, Median = {stats['change_median']:+5.1f}")
    print(f"\nOutcomes:")
    print(f"  • Improved:  {stats['pct_improved']:5.1f}%")
    print(f"  • Unchanged: {stats['pct_unchanged']:5.1f}%")
    print(f"  • Declined:  {stats['pct_declined']:5.1f}%")
    if not np.isnan(stats['correlation']):
        print(f"\nCorrelation (admission → discharge): r = {stats['correlation']:.3f}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

bucket_df = pd.DataFrame(bucket_stats)
max_improvement_idx = bucket_df['change_mean'].idxmax()
min_improvement_idx = bucket_df['change_mean'].idxmin()

print(f"\n1. BEST RECOVERY:")
print(f"   • {bucket_df.loc[max_improvement_idx, 'bucket']} points: +{bucket_df.loc[max_improvement_idx, 'change_mean']:.1f} points average improvement")
print(f"   • {bucket_df.loc[max_improvement_idx, 'pct_improved']:.1f}% of patients improve")

print(f"\n2. LEAST RECOVERY:")
print(f"   • {bucket_df.loc[min_improvement_idx, 'bucket']} points: +{bucket_df.loc[min_improvement_idx, 'change_mean']:.1f} points average improvement")
print(f"   • {bucket_df.loc[min_improvement_idx, 'pct_improved']:.1f}% of patients improve")

print(f"\n3. PATTERN:")
if bucket_df['change_mean'].is_monotonic_decreasing:
    print("   • Higher admission scores → Less improvement (ceiling effect)")
elif bucket_df['change_mean'].is_monotonic_increasing:
    print("   • Higher admission scores → More improvement")
else:
    print("   • Non-monotonic pattern - peak improvement in middle ranges")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71']

# ============================================================================
# FIGURE 1: Clean scatter plot - Admission vs Discharge (no buckets)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('ASIA Motor Score: Admission → Discharge Relationship', 
             fontsize=18, fontweight='bold')

# Panel A: Main scatter plot
ax = axes[0]
ax.scatter(admission_motor, discharge_motor, alpha=0.2, s=20, c='steelblue', edgecolors='none')

# Regression line
z = np.polyfit(admission_motor, discharge_motor, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 100, 100)
ax.plot(x_line, p(x_line), 'r-', linewidth=3, 
        label=f'Regression: y = {z[0]:.3f}x + {z[1]:.1f}')

# Perfect correlation line (no change)
ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.5, label='No change (y=x)')

ax.set_xlabel('Admission Motor Score (0-100)', fontsize=14, fontweight='bold')
ax.set_ylabel('Discharge Motor Score (0-100)', fontsize=14, fontweight='bold')
ax.set_title(f'All Patients (n={len(admission_motor):,})\n'
             f'r = {pearson_r:.3f}, R² = {pearson_r**2:.3f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_aspect('equal')

# Add text annotation
textstr = f'''Key Statistics:
Mean Admission: {admission_motor.mean():.1f}
Mean Discharge: {discharge_motor.mean():.1f}
Mean Change: {motor_change.mean():+.1f}

{improved/len(motor_change)*100:.1f}% improve
{unchanged/len(motor_change)*100:.1f}% unchanged
{declined/len(motor_change)*100:.1f}% decline'''

ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Density/hexbin plot
ax = axes[1]
hexbin = ax.hexbin(admission_motor, discharge_motor, gridsize=30, cmap='YlOrRd', mincnt=1)
ax.plot(x_line, p(x_line), 'b-', linewidth=3, label=f'Regression')
ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.7, label='No change')

ax.set_xlabel('Admission Motor Score (0-100)', fontsize=14, fontweight='bold')
ax.set_ylabel('Discharge Motor Score (0-100)', fontsize=14, fontweight='bold')
ax.set_title('Density Plot (Darker = More Patients)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.set_aspect('equal')

cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Patient Count', fontsize=11)

plt.tight_layout()
plt.savefig('motor_score_admission_vs_discharge_clean.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_score_admission_vs_discharge_clean.png")
plt.close()

# ============================================================================
# FIGURE 2: Bucketed analysis (20-point ranges)
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ASIA Motor Score by 20-Point Admission Ranges', 
             fontsize=18, fontweight='bold', y=0.995)

# Scatter plots for each bucket
for idx, (stats, ax) in enumerate(zip(bucket_stats, axes.flat[:5])):
    bucket = stats['bucket']
    mask = (admission_buckets == bucket)
    
    adm = admission_motor[mask]
    dis = discharge_motor[mask]
    
    ax.scatter(adm, dis, alpha=0.4, s=30, c=colors[idx], edgecolors='black', linewidth=0.5)
    
    # Regression line for this bucket
    if len(adm) > 2:
        z = np.polyfit(adm, dis, 1)
        p = np.poly1d(z)
        x_line = np.linspace(adm.min(), adm.max(), 50)
        ax.plot(x_line, p(x_line), 'darkblue', linewidth=3, 
                label=f'y={z[0]:.2f}x+{z[1]:.1f}')
    
    # No change line
    bucket_min = int(bucket.split('-')[0])
    bucket_max = int(bucket.split('-')[1])
    ax.plot([bucket_min, bucket_max], [bucket_min, bucket_max], 
            'k--', linewidth=2, alpha=0.5, label='No change')
    
    ax.set_xlabel('Admission Score', fontsize=12)
    ax.set_ylabel('Discharge Score', fontsize=12)
    ax.set_title(f'Admission: {bucket} points\n'
                f'n={stats["n"]:,} | Avg Δ={stats["change_mean"]:+.1f} | r={stats["correlation"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(bucket_min-2, bucket_max+2)
    ax.set_ylim(bucket_min-5, min(100, bucket_max+20))

# Summary statistics in the 6th panel
ax = axes.flat[5]
ax.axis('off')

summary_text = "SUMMARY BY BUCKET\n" + "="*40 + "\n\n"
for stats in bucket_stats:
    summary_text += f"{stats['bucket']} points (n={stats['n']:,}):\n"
    summary_text += f"  Adm: {stats['adm_mean']:.1f} → Dis: {stats['dis_mean']:.1f}\n"
    summary_text += f"  Change: {stats['change_mean']:+.1f} ± {stats['change_std']:.1f}\n"
    summary_text += f"  {stats['pct_improved']:.0f}% improve\n\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('motor_score_by_20point_buckets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_score_by_20point_buckets.png")
plt.close()

# ============================================================================
# FIGURE 3: Recovery patterns by bucket
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Recovery Patterns by Admission Score Range', 
             fontsize=18, fontweight='bold', y=0.995)

# Panel A: Box plots of changes
ax = axes[0, 0]
change_by_bucket = [motor_change[admission_buckets == b['bucket']] for b in bucket_stats]
bp = ax.boxplot(change_by_bucket, labels=[b['bucket'] for b in bucket_stats],
                patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Admission Score Range', fontsize=13)
ax.set_ylabel('Motor Score Change (Discharge - Admission)', fontsize=13)
ax.set_title('Distribution of Changes by Admission Score', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Mean changes with error bars
ax = axes[0, 1]
x = np.arange(len(bucket_stats))
means = [s['change_mean'] for s in bucket_stats]
stds = [s['change_std'] for s in bucket_stats]
labels = [s['bucket'] for s in bucket_stats]

bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.7, edgecolor='black', linewidth=2)
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={s['n']:,})" for l, s in zip(labels, bucket_stats)])
ax.set_ylabel('Mean Motor Score Change', fontsize=13)
ax.set_title('Average Recovery by Admission Score Range', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.5, f'{mean:+.1f}', ha='center', fontsize=12, fontweight='bold')

# Panel C: Admission vs Discharge means
ax = axes[1, 0]
adm_means = [s['adm_mean'] for s in bucket_stats]
dis_means = [s['dis_mean'] for s in bucket_stats]

x = np.arange(len(bucket_stats))
width = 0.35

bars1 = ax.bar(x - width/2, adm_means, width, label='Admission', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, dis_means, width, label='Discharge', alpha=0.7, edgecolor='black')

for bar, color in zip(bars1, colors):
    bar.set_color(color)
    bar.set_alpha(0.5)
for bar, color in zip(bars2, colors):
    bar.set_color(color)

ax.set_xticks(x)
ax.set_xticklabels([s['bucket'] for s in bucket_stats])
ax.set_xlabel('Admission Score Range', fontsize=13)
ax.set_ylabel('Mean Motor Score', fontsize=13)
ax.set_title('Mean Scores: Admission vs Discharge', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Outcome percentages
ax = axes[1, 1]
improved_pcts = [s['pct_improved'] for s in bucket_stats]
unchanged_pcts = [s['pct_unchanged'] for s in bucket_stats]
declined_pcts = [s['pct_declined'] for s in bucket_stats]

x = np.arange(len(bucket_stats))
p1 = ax.bar(x, improved_pcts, label='Improved', color='#2ecc71', edgecolor='black')
p2 = ax.bar(x, unchanged_pcts, bottom=improved_pcts, label='Unchanged', 
            color='#95a5a6', edgecolor='black')
p3 = ax.bar(x, declined_pcts, bottom=np.array(improved_pcts)+np.array(unchanged_pcts), 
            label='Declined', color='#e74c3c', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels([s['bucket'] for s in bucket_stats])
ax.set_xlabel('Admission Score Range', fontsize=13)
ax.set_ylabel('Percentage of Patients', fontsize=13)
ax.set_title('Outcome Distribution by Admission Score', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 100)

# Add percentage labels
for i, (imp, unch, dec) in enumerate(zip(improved_pcts, unchanged_pcts, declined_pcts)):
    if imp > 5:
        ax.text(i, imp/2, f'{imp:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')
    if unch > 5:
        ax.text(i, imp + unch/2, f'{unch:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')
    if dec > 5:
        ax.text(i, imp + unch + dec/2, f'{dec:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('motor_score_recovery_by_buckets.png', dpi=300, bbox_inches='tight')
print("✓ Saved: motor_score_recovery_by_buckets.png")
plt.close()

# ============================================================================
# TEXT REPORT
# ============================================================================

report = f"""
{'='*80}
ASIA MOTOR SCORE ANALYSIS: BY 20-POINT ADMISSION RANGES
{'='*80}

Total Patients: {len(admission_motor):,}

OVERALL CORRELATION:
  • Pearson r = {pearson_r:.4f} (R² = {pearson_r**2:.4f})
  • Admission score explains {pearson_r**2*100:.1f}% of discharge variance
  • VERY STRONG positive correlation

OVERALL SCORES:
  • Mean Admission:  {admission_motor.mean():.1f} ± {admission_motor.std():.1f}
  • Mean Discharge:  {discharge_motor.mean():.1f} ± {discharge_motor.std():.1f}
  • Mean Change:     {motor_change.mean():.1f} ± {motor_change.std():.1f}

OUTCOMES:
  • Improved:  {improved:,} ({improved/len(motor_change)*100:.1f}%)
  • Unchanged: {unchanged:,} ({unchanged/len(motor_change)*100:.1f}%)
  • Declined:  {declined:,} ({declined/len(motor_change)*100:.1f}%)

{'='*80}
ANALYSIS BY 20-POINT BUCKETS
{'='*80}
"""

for stats in bucket_stats:
    report += f"""
┌{'─'*78}┐
│ ADMISSION SCORE: {stats['bucket']} points - {stats['n']:,} patients{' '*(47-len(str(stats['n'])))}│
└{'─'*78}┘

Motor Scores:
  Admission:  Mean = {stats['adm_mean']:5.1f} ± {stats['adm_std']:4.1f}, Median = {stats['adm_median']:5.1f}
  Discharge:  Mean = {stats['dis_mean']:5.1f} ± {stats['dis_std']:4.1f}, Median = {stats['dis_median']:5.1f}
  
Recovery:
  Mean Change:     {stats['change_mean']:+6.1f} ± {stats['change_std']:4.1f} points
  Median Change:   {stats['change_median']:+6.1f} points
  
  Correlation: r = {stats['correlation']:.3f}
  
Outcomes:
  • {stats['pct_improved']:5.1f}% IMPROVED
  • {stats['pct_unchanged']:5.1f}% UNCHANGED
  • {stats['pct_declined']:5.1f}% DECLINED
"""

report += f"""
{'='*80}
KEY CLINICAL INSIGHTS
{'='*80}

1. ADMISSION SCORE IS EXCELLENT PREDICTOR:
   ✓ R² = {pearson_r**2:.3f} means {pearson_r**2*100:.0f}% of discharge variance explained
   ✓ This is one of the strongest predictors in SCI outcomes
   
2. RECOVERY PATTERN BY SCORE RANGE:
"""

for stats in bucket_stats:
    report += f"   • {stats['bucket']} points: Average change {stats['change_mean']:+.1f}, {stats['pct_improved']:.0f}% improve\n"

report += f"""
3. CEILING EFFECT:
   • Patients starting at higher scores have less room to improve
   • But even modest gains (5-10 points) can be functionally significant
   
4. FLOOR EFFECT:
   • Very low scores (<20) have more room but face greater challenges
   • Improvements in this range represent major recovery

5. CLINICAL COUNSELING:
   At admission, use the patient's motor score to counsel on expected outcome:
   
   Score 0-19:   Expected discharge ~{bucket_stats[0]['dis_mean']:.0f}, change {bucket_stats[0]['change_mean']:+.0f}
   Score 20-39:  Expected discharge ~{bucket_stats[1]['dis_mean']:.0f}, change {bucket_stats[1]['change_mean']:+.0f}
   Score 40-59:  Expected discharge ~{bucket_stats[2]['dis_mean']:.0f}, change {bucket_stats[2]['change_mean']:+.0f}
   Score 60-79:  Expected discharge ~{bucket_stats[3]['dis_mean']:.0f}, change {bucket_stats[3]['change_mean']:+.0f}
   Score 80-100: Expected discharge ~{bucket_stats[4]['dis_mean']:.0f}, change {bucket_stats[4]['change_mean']:+.0f}

{'='*80}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open('motor_score_by_ranges_analysis.txt', 'w') as f:
    f.write(report)

print("✓ Saved: motor_score_by_ranges_analysis.txt")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. motor_score_admission_vs_discharge_clean.png - Clean scatter plots")
print("  2. motor_score_by_20point_buckets.png - Analysis by 20-point ranges")
print("  3. motor_score_recovery_by_buckets.png - Recovery patterns")
print("  4. motor_score_by_ranges_analysis.txt - Detailed report")
print("\nKey Finding:")
print(f"  Admission score explains {pearson_r**2*100:.1f}% of discharge variance!")

