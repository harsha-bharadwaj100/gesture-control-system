import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "All_Datasets"
REPORT_DIR = "EDA_Reports"
SUMMARY_DIR = os.path.join(REPORT_DIR, "Summary")

# Create directories
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Feature extraction function per repetition
def extract_features(df):
    features = []
    # Group by Gesture and Repetition
    grouped = df.groupby(['Gesture_Label', 'Repetition'])
    for (gesture, rep), group in grouped:
        v = group['Voltage'].values
        if len(v) == 0: continue
        rms = np.sqrt(np.mean(v**2))
        var = np.var(v)
        mav = np.mean(np.abs(v))
        zc = np.sum(np.diff(np.signbit(v)))
        features.append({
            'Gesture': gesture,
            'Repetition': rep,
            'RMS': rms,
            'Variance': var,
            'MAV': mav,
            'Zero_Crossings': zc
        })
    return pd.DataFrame(features)

all_features = []
participant_names = []

# List all dataset files
dataset_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

for file in dataset_files:
    participant_name = os.path.basename(file).replace("5gesture_dataset.csv", "").replace(".csv", "")
    participant_names.append(participant_name)
    
    print(f"Processing {participant_name}...")
    df = pd.read_csv(file)
    
    # Ensure columns exist
    if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
        print(f"Skipping {participant_name}: Missing expected columns.")
        continue
    
    # Individual Dir
    ind_dir = os.path.join(REPORT_DIR, "Individual", participant_name)
    os.makedirs(ind_dir, exist_ok=True)
    
    # 1. Extract Features for Summary
    df_feat = extract_features(df)
    df_feat['Participant'] = participant_name
    all_features.append(df_feat)
    
    # 2. Individual Visualizations
    
    # A. Full Session Timeline (Downsampled for plotting speed)
    df_down = df.iloc[::10, :].copy()
    plt.figure(figsize=(20, 5))
    sns.lineplot(x='Timestamp', y='Voltage', hue='Gesture_Label', data=df_down, linewidth=0.5, alpha=0.8)
    plt.title(f"Full Session Timeline - {participant_name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Voltage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ind_dir, "Full_Session_Timeline.png"), dpi=150)
    plt.close()
    
    # B. Overlaid Gestures (All repetitions aligned to t=0)
    gestures = df['Gesture_Label'].unique()
    fig, axes = plt.subplots(len(gestures), 1, figsize=(12, 3 * len(gestures)), sharex=False)
    if len(gestures) == 1: axes = [axes]
    
    for ax, gesture in zip(axes, gestures):
        g_df = df[df['Gesture_Label'] == gesture]
        reps = g_df['Repetition'].unique()
        for rep in reps:
            rep_data = g_df[g_df['Repetition'] == rep]['Voltage'].values
            t_aligned = np.arange(len(rep_data)) # Time steps
            ax.plot(t_aligned, rep_data, alpha=0.3, color='blue' if gesture != 'Rest' else 'gray')
        ax.set_title(f"Gesture: {gesture} (All {len(reps)} Reps Overlaid)")
        ax.set_ylabel("Voltage")
    plt.xlabel("Time steps")
    plt.tight_layout()
    plt.savefig(os.path.join(ind_dir, "Overlaid_Gestures.png"), dpi=150)
    plt.close()
    
    # C. Individual Feature Distributions
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gesture', y='RMS', data=df_feat)
    plt.title(f"RMS Distribution by Gesture - {participant_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ind_dir, "RMS_Boxplot.png"))
    plt.close()

# --- SUMMARY VISUALIZATIONS ---
print("Generating Summary Visualizations...")
if len(all_features) > 0:
    global_feat_df = pd.concat(all_features, ignore_index=True)
    
    # Global Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Gesture', y='RMS', hue='Participant', data=global_feat_df)
    plt.title("Global RMS Distribution by Gesture across Participants")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Global_RMS_Boxplot.png"), dpi=150)
    plt.close()
    
    # Global Boxplot (Grouped by Gesture only, ignoring participant)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gesture', y='RMS', data=global_feat_df, inner="quartile")
    plt.title("Overall RMS Distribution by Gesture (Violin)")
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "Global_RMS_Violin.png"), dpi=150)
    plt.close()
    
    # PCA / t-SNE
    features = ['RMS', 'Variance', 'MAV', 'Zero_Crossings']
    X = global_feat_df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    global_feat_df['PCA1'] = X_pca[:, 0]
    global_feat_df['PCA2'] = X_pca[:, 1]
    
    # PCA colored by Gesture
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Gesture', style='Participant', data=global_feat_df, alpha=0.7)
    plt.title("PCA of Extracted Features (Colored by Gesture)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "PCA_Gestures.png"), dpi=150)
    plt.close()

print("EDA Generation Complete. Check the EDA_Reports directory.")
