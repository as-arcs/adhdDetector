import os
import subprocess

folder = "models"

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".py"):
        script_path = os.path.join(folder, filename)
        print(f"\nRunning {filename}...")

        subprocess.run(["python", script_path], check=True)