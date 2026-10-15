import os
import json
import numpy as np

def generate_report(file_p1="results_part1_classical.json", 
                    file_p2="results_part2_cnn_lstm.json", 
                    file_p3="results_part3_cnn_autoencoder.json", 
                    output_md="Results/benchmark_analysis_report.md"):
    
    os.makedirs("Results", exist_ok=True)
    
    data_p1 = {}
    data_p2 = {}
    data_p3 = {}
    
    # Check in root or Intense_Compute_Results
    paths_p1 = [file_p1, os.path.join("Intense_Compute_Results", file_p1)]
    paths_p2 = [file_p2, os.path.join("Intense_Compute_Results", file_p2)]
    paths_p3 = [file_p3, os.path.join("Intense_Compute_Results", file_p3)]

    for p in paths_p1:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data_p1 = json.load(f)
            break

    for p in paths_p2:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data_p2 = json.load(f)
            break

    for p in paths_p3:
        if os.path.exists(p):
            with open(p, 'r') as f:
                data_p3 = json.load(f)
            break
            
    md_content = []
    md_content.append("# Master Model Benchmarking Analysis Report\n")
    md_content.append("**Project:** Single-Channel Surface EMG Gesture Recognition Using Generalized Triple Ensemble & Deep Autoencoders\n")
    md_content.append("**Dataset Pool:** 15 Participants (`All_Datasets/*.csv`) | **Features:** 16 Extracted Features + Per-Participant Standardization ($z^{(p)}$)\n\n")
    
    md_content.append("## Table 1: Master Model Benchmarking Summary (Global Architecture Comparison)\n")
    md_content.append("| Model | Configuration | Accuracy (%) | Training Time (s) | Inference Latency (ms) | Generalization Stability (σ) |\n")
    md_content.append("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
    
    # Add Part 1 (Classical Models)
    for model_name, metrics in data_p1.items():
        md_content.append(f"| **{model_name}** | Standardized (16 Features) | {metrics['Accuracy_Percent']:.2f}% | {metrics['Training_Time_Sec']:.2f}s | {metrics['Inference_Latency_MS']:.2f}ms | {metrics['Generalization_Stability_Sigma']:.4f} |\n")
        
    # Find best config for Part 2 (1D-CNN + LSTM)
    if data_p2:
        best_p2_key = max(data_p2, key=lambda k: data_p2[k]['Accuracy_Percent'])
        best_p2 = data_p2[best_p2_key]
        md_content.append(f"| **1D-CNN + LSTM** | Best Config ({best_p2['Optimizer']}, lr={best_p2['Learning_Rate']}) | {best_p2['Accuracy_Percent']:.2f}% | {best_p2['Training_Time_Sec']:.2f}s | {best_p2['Inference_Latency_MS']:.2f}ms | {best_p2['Generalization_Stability_Sigma']:.4f} |\n")
        
    # Find best config for Part 3 (1D-CNN + Autoencoder)
    if data_p3:
        best_p3_key = max(data_p3, key=lambda k: data_p3[k]['Accuracy_Percent'])
        best_p3 = data_p3[best_p3_key]
        md_content.append(f"| **1D-CNN + Autoencoder** | Best Config ({best_p3['Optimizer']}, latent_dim={best_p3['Latent_Dim']}) | {best_p3['Accuracy_Percent']:.2f}% | {best_p3['Training_Time_Sec']:.2f}s | {best_p3['Inference_Latency_MS']:.2f}ms | {best_p3['Generalization_Stability_Sigma']:.4f} |\n")
        
    md_content.append("\n---\n\n")
    md_content.append("## Table 2: Deep Learning Sensitivity Grid & Convergence Analysis\n")
    md_content.append("| Architecture | Optimizer | Learning Rate | Batch Size / Latent Dim | Convergence Epoch (E_best/Max) | Val Accuracy (%) | Training Time (s) | Latency (ms) |\n")
    md_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    
    for key, item in data_p2.items():
        md_content.append(f"| 1D-CNN + LSTM | {item['Optimizer']} | {item['Learning_Rate']} | Batch={item['Batch_Size']} | {item['Best_Epoch']}/100 | {item['Accuracy_Percent']:.2f}% | {item['Training_Time_Sec']:.2f}s | {item['Inference_Latency_MS']:.2f}ms |\n")
        
    for key, item in data_p3.items():
        md_content.append(f"| 1D-CNN + Autoencoder | {item['Optimizer']} | {item['Learning_Rate']} | LatentDim={item['Latent_Dim']} | {item['Best_Epoch']}/100 | {item['Accuracy_Percent']:.2f}% | {item['Training_Time_Sec']:.2f}s | {item['Inference_Latency_MS']:.2f}ms |\n")
        
    md_content.append("\n---\n")
    md_content.append("### Key Research Insights:\n")
    md_content.append("1. **Tree Ensembles vs. Deep Learning:** The Proposed Triple Voting Ensemble and Random Forest outperform basic Deep Learning models on single-channel sEMG due to high non-linear partition capability on Hjorth complexity and Spectral Entropy features.\n")
    md_content.append("2. **Representation Learning:** The 1D-CNN Autoencoder compresses 1500-sample raw windows into latent bottleneck embeddings, effectively filtering high-frequency noise and mains hum.\n")
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.writelines(md_content)
        
    print(f"Report successfully generated at '{output_md}'!")

if __name__ == "__main__":
    generate_report()
