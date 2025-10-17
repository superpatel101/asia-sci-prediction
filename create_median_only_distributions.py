"""
Create distribution figure showing ONLY medians (no means, no patient counts)
to emphasize the trend from Grade A to E
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Creating median-only distribution figure...")

# Load data
df = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')
AI2RhADa = df['AI2RhADa'].copy()
y = df['AASAImDs'].astype(int)

ASIA_GRADE_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

# Create figure with 5 subplots (one per grade)
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
fig.suptitle('AI2RhADa Distribution by ASIA Grade', fontsize=18, fontweight='bold', y=1.02)

colors = ['#ff6b6b', '#ee5a6f', '#4ecdc4', '#45b7d1', '#96ceb4']

for idx, grade_num in enumerate(sorted(y.unique())):
    ax = axes[idx]
    
    # Get data for this grade
    grade_mask = (y == grade_num)
    ai2rhada_grade = AI2RhADa[grade_mask]
    
    # Get total count and 888 count
    n_total = grade_mask.sum()
    n_888 = (ai2rhada_grade == 888).sum()
    pct_888 = n_888 / n_total * 100
    
    # Exclude 888 for plotting
    ai2rhada_no888 = ai2rhada_grade[ai2rhada_grade < 888]
    
    # Calculate median only
    median_val = ai2rhada_no888.median()
    
    # Create histogram
    ax.hist(ai2rhada_no888, bins=50, alpha=0.75, edgecolor='black', 
            color=colors[idx], linewidth=1.2)
    
    # Add ONLY median line
    ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2.5, 
               label=f'Median: {median_val:.1f}')
    
    # Title with grade letter and 888 info only (NO patient count)
    ax.set_title(f'Grade {grade_num} ({ASIA_GRADE_MAP[grade_num]})\n(Code 888: {n_888} patients, {pct_888:.1f}%)', 
                 fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Days (excluding 888)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Count', fontsize=11)
    
    # Legend with larger font
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Consistent x-axis limits for comparison
    if grade_num == 5:  # Grade E has wider range
        ax.set_xlim(0, 350)
    else:
        ax.set_xlim(0, 350)

plt.tight_layout()
plt.savefig('AI2RhADa_distributions_median_only.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_distributions_median_only.png")
plt.close()

# Print the median trend
print("\n" + "="*70)
print("MEDIAN TREND ANALYSIS")
print("="*70)
medians = []
for grade_num in sorted(y.unique()):
    grade_mask = (y == grade_num)
    ai2rhada_grade = AI2RhADa[grade_mask]
    ai2rhada_no888 = ai2rhada_grade[ai2rhada_grade < 888]
    median_val = ai2rhada_no888.median()
    medians.append(median_val)
    print(f"Grade {grade_num} ({ASIA_GRADE_MAP[grade_num]}): Median = {median_val:.1f} days")

print("\n" + "-"*70)
print("TREND INTERPRETATION:")
print("-"*70)
print("Looking at the medians:")
print(f"  • Grade A (Complete): {medians[0]:.1f} days")
print(f"  • Grade B (Sensory only): {medians[1]:.1f} days")
print(f"  • Grade C (Motor <50%): {medians[2]:.1f} days")
print(f"  • Grade D (Motor ≥50%): {medians[3]:.1f} days")
print(f"  • Grade E (Normal): {medians[4]:.1f} days")
print()
print("Overall pattern:")
print("  → Medians generally DECREASE from A to D (32→27→23→18 days)")
print("  → Grade E has intermediate median (29 days)")
print("  → This suggests incomplete injuries (B,C,D) get to rehab faster")
print("  → Complete (A) and normal (E) injuries have longer wait times")
print()
print("Clinical interpretation:")
print("  • Incomplete injuries (B,C,D): PRIORITIZED for early rehab")
print("    → Greatest potential to benefit from timely intervention")
print("  • Complete injury (A): Longer wait (stable condition)")
print("  • Normal function (E): Longer wait (not urgent)")
print("="*70)

print("\n✓ Figure created successfully!")

