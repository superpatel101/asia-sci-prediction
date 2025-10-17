"""
Create distribution figure showing ONLY medians with proper 3-over-2 layout
No 888 counts in titles - explanation in bottom right
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Creating median-only distribution figure with 888 explanation...")

# Load data
df = pd.read_csv('/Users/aaryanpatel/Downloads/ModelreadyAISMedsurgtodischarge.csv')
AI2RhADa = df['AI2RhADa'].copy()
y = df['AASAImDs'].astype(int)

ASIA_GRADE_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

# Create figure with 2 rows: 3 plots top, 2 plots bottom
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

fig.suptitle('AI2RhADa Distribution by ASIA Grade', fontsize=20, fontweight='bold', y=0.98)

colors = ['#ff6b6b', '#ee5a6f', '#4ecdc4', '#45b7d1', '#96ceb4']

# Define subplot positions
# First row: Grades 1, 2, 3
# Second row: Grades 4, 5 (centered by using columns 0-1 and 1-2)
subplot_positions = [
    (0, 0),  # Grade 1 (A) - row 0, col 0
    (0, 1),  # Grade 2 (B) - row 0, col 1
    (0, 2),  # Grade 3 (C) - row 0, col 2
    (1, 0),  # Grade 4 (D) - row 1, col 0
    (1, 1),  # Grade 5 (E) - row 1, col 1
]

grades = sorted(y.unique())

for idx, grade_num in enumerate(grades):
    row, col = subplot_positions[idx]
    ax = fig.add_subplot(gs[row, col])
    
    # Get data for this grade
    grade_mask = (y == grade_num)
    ai2rhada_grade = AI2RhADa[grade_mask]
    
    # Exclude 888 for plotting
    ai2rhada_no888 = ai2rhada_grade[ai2rhada_grade < 888]
    
    # Calculate median only
    median_val = ai2rhada_no888.median()
    
    # Determine number of bins based on sample size and range
    if grade_num == 4:  # Grade D
        bins = 15
    elif grade_num == 5:  # Grade E
        bins = 60
    elif grade_num == 3:  # Grade C
        bins = 50
    else:  # Grades A, B
        bins = 45
    
    # Create histogram
    ax.hist(ai2rhada_no888, bins=bins, alpha=0.75, edgecolor='black', 
            color=colors[idx], linewidth=1.2)
    
    # Add ONLY median line
    ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=3, 
               label=f'Median: {median_val:.1f}', zorder=10)
    
    # Simple title - just grade letter (NO 888 info)
    ax.set_title(f'Grade {grade_num} ({ASIA_GRADE_MAP[grade_num]})', 
                 fontsize=14, fontweight='bold', pad=12)
    
    # Labels
    ax.set_xlabel('Days (excluding 888)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    # Legend
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set consistent x-axis limits
    ax.set_xlim(0, 350)
    
    # Make tick labels slightly larger
    ax.tick_params(axis='both', labelsize=10)

# Add explanation text in the bottom right subplot
ax_text = fig.add_subplot(gs[1, 2])
ax_text.axis('off')

explanation_text = """
Code 888 Explanation

AI2RhADa = Days from Injury to 
Rehabilitation Admission

Code 888 means:
"Not Applicable - Patient was not 
admitted to System inpatient 
Rehabilitation"

These patients are excluded from 
the histograms above to show only 
valid time-to-rehab values.

Note: Grade D has the highest 
percentage of code 888 (62.9%), 
while other grades range from 
1.6% to 2.8%.
"""

ax_text.text(0.1, 0.5, explanation_text, 
            fontsize=12, 
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1),
            family='monospace',
            linespacing=1.8)

plt.savefig('AI2RhADa_distributions_median_only.png', dpi=300, bbox_inches='tight')
print("✓ Saved: AI2RhADa_distributions_median_only.png")
plt.close()

# Print the median trend
print("\n" + "="*70)
print("MEDIAN TREND ANALYSIS")
print("="*70)
for grade_num in sorted(y.unique()):
    grade_mask = (y == grade_num)
    ai2rhada_grade = AI2RhADa[grade_mask]
    ai2rhada_no888 = ai2rhada_grade[ai2rhada_grade < 888]
    median_val = ai2rhada_no888.median()
    print(f"Grade {grade_num} ({ASIA_GRADE_MAP[grade_num]}): Median = {median_val:.1f} days")

print("\n✓ Figure created with 888 explanation in bottom right!")

