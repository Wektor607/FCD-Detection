import os
import pandas as pd
import sys
meld_dir = "/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/meld_files"
csv_path = "/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/meld_files/all_augmented_reports.csv"

# 1) все control папки
all_controls = {d for d in os.listdir(meld_dir) if os.path.isdir(os.path.join(meld_dir, d)) and "_C_" in d}

# 2) все subject_id в CSV
df = pd.read_csv(csv_path)
processed = set(df["subject_id"].astype(str))
print(processed)
# 3) недостающие controls
missing_controls = sorted(all_controls - processed)
print("Missing:", missing_controls)
sys.exit(0)
# 4) дозаписываем
with open(csv_path, "a", encoding="utf8") as f:
    for sid in missing_controls:
        f.write(f"{sid},No lesion detected,No lesion detected\n")

print("Done.")
