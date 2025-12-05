"""
Some annotation programs output random IDs for their classes, this script will enumerate them from zero.
"""

from shared import *
import json

with open(ANNOTATIONS_COCO_FILE) as f:
    data = json.load(f)

# Extract unique original category IDs
original_ids = sorted({cat["id"] for cat in data["categories"]})
# Create a mapping: original ID -> new sequential ID starting at 1
id_map = {old_id: new_id for new_id, old_id in enumerate(original_ids, start=1)}

print("Class mapping (src -> target): ", id_map)

# Update category IDs in categories
for cat in data["categories"]:
    cat["id"] = id_map[cat["id"]]

# Update category IDs in annotations
for ann in data["annotations"]:
    ann["category_id"] = id_map[ann["category_id"]]

# Save the fixed JSON file
with open(ANNOTATIONS_COCO_FIXED_FILE, "w") as f:
    json.dump(data, f)
