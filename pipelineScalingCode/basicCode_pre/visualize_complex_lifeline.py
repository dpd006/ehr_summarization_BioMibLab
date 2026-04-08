import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1. Setup Data Path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading data...")
# Load Admissions and Diagnoses
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')

# Merge diagnosis names so we can read them
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='inner')

# Convert dates to actual datetime objects
df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
df_adm['dischtime'] = pd.to_datetime(df_adm['dischtime'])

# ---------------------------------------------------------
# 2. CONFIGURATION: Patient 10015860 (The Complex Case)
# ---------------------------------------------------------
patient_id = 10015860
print(f"Filtering data for Patient {patient_id}...")

subset_adm = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')

if subset_adm.empty:
    print("Error: Patient not found!")
    exit()

# ---------------------------------------------------------
# 3. PLOTTING THE LIFELINE
# ---------------------------------------------------------
# Make the chart wide (15 inches) to fit all 13 visits
fig, ax = plt.subplots(figsize=(15, 6))

# Define Y-position for the main timeline
y_level = 1 

# Draw the background "Lifeline" (Gray line from first admit to last discharge)
start_overall = subset_adm['admittime'].min()
end_overall = subset_adm['dischtime'].max()
ax.plot([start_overall, end_overall], [y_level, y_level], color='gray', alpha=0.3, linewidth=2, zorder=1)

print(f"Found {len(subset_adm)} visits. generating plot...")

# Loop through every admission to draw the Blue Bars
for i, (_, row) in enumerate(subset_adm.iterrows()):
    start = row['admittime']
    end = row['dischtime']
    hadm_id = row['hadm_id']
    
    # Get the top diagnosis for the label
    # We grab the first one listed for this admission
    diags = df_diag[df_diag['hadm_id'] == hadm_id]['long_title'].values
    if len(diags) > 0:
        # Pick a diagnosis that isn't just "Hypertension" if possible, to be interesting
        # (Simple logic: just take the first one for now)
        label_text = diags[0]
        # Truncate text if it's too long (e.g. > 20 chars)
        if len(label_text) > 20:
            label_text = label_text[:20] + "..."
    else:
        label_text = "Unknown"

    # Draw the Hospital Stay (Blue Bar)
    ax.plot([start, end], [y_level, y_level], color='#007acc', linewidth=12, solid_capstyle='round', zorder=2)
    
    # Add Text Label (Stagger them up and down so they don't overlap)
    # Even numbers go Up, Odd numbers go Down
    text_y = y_level + 0.03 if i % 2 == 0 else y_level - 0.05
    vertical_align = 'bottom' if i % 2 == 0 else 'top'
    
    ax.text(start, text_y, f"Visit {i+1}\n{label_text}", 
            fontsize=8, rotation=45, ha='left', va=vertical_align, color='#333333')

    # Draw Red "Gap" info (Time at home)
    if i > 0:
        prev_end = subset_adm.iloc[i-1]['dischtime']
        gap_days = (start - prev_end).days
        
        # Only label gaps if they are big enough to see (> 30 days)
        if gap_days > 30:
            mid_point = prev_end + (start - prev_end)/2
            ax.text(mid_point, y_level, f"{gap_days} days", 
                    ha='center', va='center', fontsize=7, color='red', backgroundcolor='white', zorder=3)

# ---------------------------------------------------------
# 4. FORMATTING
# ---------------------------------------------------------
ax.set_yticks([]) # Hide Y axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Format X-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

ax.set_title(f"Patient {patient_id} Lifeline: 13 Visits over 7 Years", fontsize=14, fontweight='bold')
ax.set_ylim(0.8, 1.3) # Zoom in vertically

plt.tight_layout()

# Save it
filename = f"lifeline_{patient_id}.png"
plt.savefig(filename, dpi=150)
print(f"\n[SUCCESS] Image saved as: {filename}")
print("Go open this image to see the progression!")