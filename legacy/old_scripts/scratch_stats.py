import pandas as pd
import numpy as np

df = pd.read_csv('All_Datasets/harsha5gesture_dataset.csv')
df['Time_Idx'] = df.groupby(['Gesture_Label', 'Repetition']).cumcount()
profile = df.groupby('Time_Idx')['Voltage'].apply(lambda x: np.mean(np.abs(x))).reset_index()

for idx in [0, 100, 200, 300, 400, 500, 600, 1000, 2000, 3000, 3900]:
    row = profile[profile['Time_Idx'] == idx]
    if not row.empty:
        val = row["Voltage"].values[0]
        print(f"Sample {idx:4d} ({idx}ms): Average MA Voltage = {val:.6f}")
