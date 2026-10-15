import os
import shutil
import zipfile

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
