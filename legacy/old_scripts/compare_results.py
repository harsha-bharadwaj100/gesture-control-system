import os
import glob
import json
import pandas as pd

RESULTS_DIR = "Results"

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: {RESULTS_DIR} directory not found. Please run the individual model scripts first.")
        return
        
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*_result.json"))
    
    if len(result_files) == 0:
        print("No result files found in the Results directory.")
        return
        
    results = []
    
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
            
    df = pd.DataFrame(results)
    
    # Sort by accuracy descending
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    print("\n==============================")
    print("FINAL MODEL COMPARISON RESULTS")
    print("==============================\n")
    print(df.to_string())
    
    # Optionally save the comparison to a CSV
    df.to_csv(os.path.join(RESULTS_DIR, "final_comparison.csv"), index=False)
    print(f"\nSaved final comparison to {os.path.join(RESULTS_DIR, 'final_comparison.csv')}")

if __name__ == "__main__":
    main()
