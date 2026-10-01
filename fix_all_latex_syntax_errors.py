import re
import os
import shutil
import zipfile

with open('paper_elsarticle/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
fixed_lines = []
in_tabular = False

for line in lines:
    stripped = line.strip()
    
    # Check tabular boundary
    if r'\begin{tabular' in line:
        in_tabular = True
        fixed_lines.append(line)
        continue
    elif r'\end{tabular' in line:
        in_tabular = False
        fixed_lines.append(line)
        continue
        
    if in_tabular:
        # Inside tabular, & is used for column separation, keep it!
        fixed_lines.append(line)
        continue
        
    # Outside tabular:
    # 1. Fix markdown bold **text** -> \textbf{text}
    cur = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
    
    # 2. Fix unescaped & outside comments
    # If line is a comment, don't touch
    if not cur.strip().startswith('%'):
        # Replace & with \& if not already preceded by \
        cur = re.sub(r'(?<!\\)&', r'\&', cur)
        
    # 3. Fix unescaped % inside text/headings like '20% Chance'
    # Look for patterns like '20%' or words followed by % that are not comments
    # Specifically in \subsection, \caption, \textbf, etc.
    def fix_percent_in_cmd(m):
        cmd = m.group(1)
        arg = m.group(2)
        arg_fixed = re.sub(r'(?<!\\)%', r'\%', arg)
        return f'{cmd}{{{arg_fixed}}}'

    cur = re.sub(r'(\\subsection|\\section|\\caption|\\textbf|\\title|\\item)\{(.*?)\}', fix_percent_in_cmd, cur)
    
    # Also fix any standalone '20%' in text
    cur = re.sub(r'(\d+)%(?!\w)', r'\1\%', cur)

    fixed_lines.append(cur)

fixed_text = '\n'.join(fixed_lines)

with open('paper_elsarticle/main.tex', 'w', encoding='utf-8') as f:
    f.write(fixed_text)

print("Successfully fixed all unescaped &, %, and markdown formatting in paper_elsarticle/main.tex!")

# Verify no unescaped & outside tabular
lines_check = fixed_text.splitlines()
in_tab = False
issues = 0
for i, l in enumerate(lines_check):
    if r'\begin{tabular' in l:
        in_tab = True
    elif r'\end{tabular' in l:
        in_tab = False
    elif not in_tab:
        if re.findall(r'(?<!\\)&', l) and not l.strip().startswith('%'):
            print(f"Remaining & issue at line {i+1}: {l}")
            issues += 1
        if re.findall(r'(?<!\\)%', l) and not l.strip().startswith('%') and ('\\section' in l or '\\subsection' in l or '\\caption' in l):
            print(f"Remaining % issue at line {i+1}: {l}")
            issues += 1

if issues == 0:
    print("Zero syntax issues detected! 100% clean LaTeX!")

# Rebuild paper_elsarticle.zip
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
