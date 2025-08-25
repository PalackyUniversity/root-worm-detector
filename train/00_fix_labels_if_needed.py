import json

JSON_INPUT = "sliced/sliced_output.json_coco.json"
JSON_OUTPUT = "sliced/fixed_sliced_output.json"

with open(JSON_INPUT) as f:
    data = json.load(f)

# Extract unique original category IDs
original_ids = sorted({cat["id"] for cat in data["categories"]})

# Create a mapping: original ID -> new sequential ID starting at 1
id_map = {old_id: new_id for new_id, old_id in enumerate(original_ids, start=1)}

# Update category IDs in categories
for cat in data["categories"]:
    cat["id"] = id_map[cat["id"]]

# Update category IDs in annotations
for ann in data["annotations"]:
    ann["category_id"] = id_map[ann["category_id"]]

# Save the fixed JSON file
with open(JSON_OUTPUT, "w") as f:
    json.dump(data, f)
