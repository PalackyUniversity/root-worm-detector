from tqdm import tqdm
import random
import shutil
import os

# Paths
FOLDER_DATA = "sliced"
FOLDER_LABELS = "coco_to_yolo_output/labels/fixed_sliced_output"
FOLDER_VAL_IMAGES = "images/val"
FOLDER_VAL_LABELS = "labels/val"
FOLDER_TRAIN_IMAGES = "images/train"
FOLDER_TRAIN_LABELS = "labels/train"

THRESHOLD = 0.7

# Create output directories if they don't exist
os.makedirs(FOLDER_TRAIN_IMAGES, exist_ok=True)
os.makedirs(FOLDER_VAL_IMAGES, exist_ok=True)
os.makedirs(FOLDER_TRAIN_LABELS, exist_ok=True)
os.makedirs(FOLDER_VAL_LABELS, exist_ok=True)

# Process label files
for label_file in tqdm(os.listdir(FOLDER_LABELS)):
    label_path = os.path.join(FOLDER_LABELS, label_file)

    # Assume image has same name but in data folder
    image_file = label_file.rsplit(".", 1)[0] + ".png"
    image_path = os.path.join(FOLDER_DATA, image_file)

    if os.path.exists(image_path):
        if random.random() < THRESHOLD:
            shutil.copy2(image_path, FOLDER_TRAIN_IMAGES)
            shutil.copy2(label_path, FOLDER_TRAIN_LABELS)
        else:
            shutil.copy2(image_path, FOLDER_VAL_IMAGES)
            shutil.copy2(label_path, FOLDER_VAL_LABELS)
    else:
        print(f"Image not found: {image_path}")
