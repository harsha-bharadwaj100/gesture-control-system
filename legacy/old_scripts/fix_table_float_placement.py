import os
import shutil
import zipfile

with open('paper_elsarticle/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update preamble with float controls and placeins
target_pkg = r"\usepackage{amsmath,amssymb,graphicx,booktabs,array,hyperref,subcaption,url,adjustbox}"
replacement_pkg = r"""\usepackage{amsmath,amssymb,graphicx,booktabs,array,hyperref,subcaption,url,adjustbox}
\usepackage{float}
\usepackage{placeins}

% Float control parameters to prevent tables from being pushed to the end
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.80}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.80}"""

if target_pkg in text:
    text = text.replace(target_pkg, replacement_pkg, 1)
    print("Added float packages and relaxed float fractions in preamble!")

# 2. Update Table 1: [t] -> [!t]
text = text.replace(r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\caption{Systematic Comparison",
                    r"\begin{table*}[!t]" + "\n" + r"\centering" + "\n" + r"\caption{Systematic Comparison")

# 3. Update Table 2: [h] -> [!htbp]
text = text.replace(r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\caption{The 16 sEMG Feature Set",
                    r"\begin{table}[!htbp]" + "\n" + r"\centering" + "\n" + r"\caption{The 16 sEMG Feature Set")

# 4. Update Table 3: [h] -> [!htbp]
text = text.replace(r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\caption{Table 3: Master Model Benchmarking Summary}",
                    r"\begin{table}[!htbp]" + "\n" + r"\centering" + "\n" + r"\caption{Table 3: Master Model Benchmarking Summary}")

# 5. Update Table 4: [t] -> [!t] and add \small before tabular so vertical height is reduced
t4_old = r"""\begin{table*}[t]
\centering
\caption{Table 4: Deep Learning Sensitivity Grid \& Convergence Analysis (Complete 42-Row Hyperparameter Exploration)}
\label{tab:sensitivity_grid_full}
\resizebox{\textwidth}{!}{%"""

t4_new = r"""\begin{table*}[!t]
\centering
\small
\caption{Table 4: Deep Learning Sensitivity Grid \& Convergence Analysis (Complete 42-Row Hyperparameter Exploration)}
\label{tab:sensitivity_grid_full}
\resizebox{\textwidth}{!}{%"""

if t4_old in text:
    text = text.replace(t4_old, t4_new)
    print("Updated Table 4 to [!t] with \\small height compression!")

# 6. Add \FloatBarrier before Section 8
s8_target = r"\section{Subject-Wise Generalization \& Statistical Hypothesis Testing}"
if r"\FloatBarrier" not in text:
    text = text.replace(s8_target, r"\FloatBarrier" + "\n\n" + s8_target)
    print("Added \\FloatBarrier before Section 8!")

# 7. Update Table 5: [h] -> [!htbp]
t5_old = r"""\begin{table}[h]
\centering
\caption{Table 5: Subject-Wise Classification Accuracy and Statistical Metrics ($N=15$)}"""

t5_new = r"""\begin{table}[!htbp]
\centering
\caption{Table 5: Subject-Wise Classification Accuracy and Statistical Metrics ($N=15$)}"""

if t5_old in text:
    text = text.replace(t5_old, t5_new)
    print("Updated Table 5 to [!htbp]!")

with open('paper_elsarticle/main.tex', 'w', encoding='utf-8') as f:
    f.write(text)

# 8. Rebuild paper_elsarticle.zip
zip_filename = 'paper_elsarticle.zip'
print(f"Packaging {zip_filename}...")

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('paper_elsarticle'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'paper_elsarticle')
            zipf.write(file_path, arcname)
            print(f" Added: {arcname}")

artifact_dir = r'C:\Users\harsh\.gemini\antigravity\brain\81964831-9574-4fb0-948b-c810cc4ca936'
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy(zip_filename, os.path.join(artifact_dir, zip_filename))
print("Successfully generated and copied clean paper_elsarticle.zip!")
