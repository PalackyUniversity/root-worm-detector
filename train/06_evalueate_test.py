from glob import glob
import pandas as pd
import pickle
import os

folders = ["25_06_16_infekce", "25_06_23_infekce", "25_06_30_infekce"]
all_data = {}

for folder in folders:
    folder_results = {}
    for filepath in glob(f"{folder}/pickles/*.pickle"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        count_small = 0
        for obj_pred in data:
            bbox = obj_pred.bbox
            width = bbox.maxx - bbox.minx
            height = bbox.maxy - bbox.miny
            area = width * height

            if obj_pred.score.is_greater_than_threshold(0.2):
                count_small += 1

        filename = "_".join(os.path.basename(filepath).split("_")[:-1]).replace("-_", "_").replace("-", "_").replace("EL47", "EL_47").replace("F_DMSO_100", "F_DMSO")
        folder_results[filename] = count_small

    all_data[folder] = folder_results

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Sort rows by filename
df = df.sort_index()

# Save final CSV
df.to_csv("test/combined_results.csv", index_label="filename")