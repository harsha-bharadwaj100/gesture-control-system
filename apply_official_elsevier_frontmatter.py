import os
import shutil
import zipfile

# 1. Ensure official elsarticle.cls is in paper_elsarticle
shutil.copy('test_extracted_elsarticle.cls', 'paper_elsarticle/elsarticle.cls')
print("Installed official 1438-line elsarticle.cls v3.5 to paper_elsarticle/elsarticle.cls")

# 2. Copy official BST file if available
bst_src = r'C:\Users\harsh\Downloads\fyp\elsarticle\elsarticle\elsarticle-num.bst'
if os.path.exists(bst_src):
    shutil.copy(bst_src, 'paper_elsarticle/elsarticle-num.bst')
    print("Copied official elsarticle-num.bst to paper_elsarticle/")

# 3. Read current main.tex
with open('paper_elsarticle/main.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Replace documentclass and frontmatter with exact official Elsevier structure
# Author list:
# 1. Priyanka S. (Assistant Professor, CSE)
# 2. Arpita K. (Assistant Professor, ECE)
# 3. Harsha Bharadwaj (CSE, Corresponding Author)
# 4. Guru Raj (CSE)
# 5. Keerthana K. (CSE)

new_frontmatter = r"""\documentclass[5p,times,twocolumn]{elsarticle}

\usepackage{amsmath,amssymb,graphicx,booktabs,array,hyperref,subcaption,url,adjustbox}

\journal{Biomedical Signal Processing and Control}

\begin{document}

\begin{frontmatter}

\title{A Generalized Two-Electrode Bipolar Differential sEMG Sign-to-Speech Assistive Interface Using Per-Participant Standardization and Multi-Domain Ensembles}

\author[cse]{Priyanka S.}
\author[ece]{Arpita K.}
\author[cse]{Harsha Bharadwaj\corref{cor1}}
\ead{harshabharadwaj100@gmail.com}
\author[cse]{Guru Raj}
\author[cse]{Keerthana K.}

\affiliation[cse]{organization={Department of Computer Science and Engineering, BNM Institute of Technology},
            city={Bangalore},
            country={India}}

\affiliation[ece]{organization={Department of Electronics and Communication Engineering, BNM Institute of Technology},
            city={Bangalore},
            country={India}}

\cortext[cor1]{Corresponding author}"""

# Find where abstract starts
abs_start_idx = tex.find(r'\begin{abstract}')
assert abs_start_idx != -1, "Could not find begin abstract!"

# Cut out old preamble and frontmatter up to begin{abstract}
updated_tex = new_frontmatter + "\n\n" + tex[abs_start_idx:]

with open('paper_elsarticle/main.tex', 'w', encoding='utf-8') as f:
    f.write(updated_tex)

print("Successfully updated paper_elsarticle/main.tex with official Elsevier frontmatter and updated author list!")

# 4. Rebuild paper_elsarticle.zip
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
print("Successfully generated and copied paper_elsarticle.zip!")
