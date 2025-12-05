from glob import glob
import pandas as pd
import pickle
import os

all_data = {}

for filepath in glob(f"runs/predict/exp/pickles/*.pickle"):
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

    all_data[".".join(os.path.basename(filepath).split(".")[:-1])] = count_small

# Convert to DataFrame
df = pd.DataFrame({"counts": all_data})

# Sort rows by filename
df = df.sort_index()

# Save final CSV
df.to_csv("combined_results.csv", index_label="filename")
