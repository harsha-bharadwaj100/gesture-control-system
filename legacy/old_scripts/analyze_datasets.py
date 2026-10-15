import pandas as pd
import glob
import os

files = glob.glob('All_Datasets/*.csv')
all_stats = []
for f in files:
    df = pd.read_csv(f)
    stats = {'File': os.path.basename(f), 'Rows': len(df), 'Columns': len(df.columns)}
    label_cols = [c for c in df.columns if 'label' in c.lower() or 'gesture' in c.lower() or 'class' in c.lower()]
    if label_cols:
        col = label_cols[0]
        stats['Classes'] = df[col].nunique()
        stats['Class_Distribution'] = df[col].value_counts().to_dict()
    else:
        stats['Classes'] = 0
    stats['Missing'] = df.isna().sum().sum()
    all_stats.append(stats)
    
print("--- Overall Summary ---")
df_stats = pd.DataFrame(all_stats)
print(df_stats[['File', 'Rows', 'Columns', 'Classes', 'Missing']])

print("\n--- Example Column Schema ---")
df_first = pd.read_csv(files[0])
print(list(df_first.columns))

print("\n--- Class Distributions ---")
for s in all_stats:
    print(f"{s['File']}: {s.get('Class_Distribution', 'N/A')}")
