from shared_paths import *
from tqdm import tqdm
import random
import shutil
import os

THRESHOLD = 0.8

# Create output directories if they don't exist
os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_TRAIN_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_VAL_DIR, exist_ok=True)

# Process label files
for label_file in tqdm(os.listdir(FOLDER_LABELS)):
    label_path = os.path.join(FOLDER_LABELS, label_file)

    # Assume image has same name but in data folder
    image_file = label_file.rsplit(".", 1)[0] + ".png"
    image_path = os.path.join(IMAGES_SLICED_DIR, image_file)

    if os.path.exists(image_path):
        if random.random() < THRESHOLD:
            shutil.copy2(image_path, IMAGES_TRAIN_DIR)
            shutil.copy2(label_path, ANNOTATIONS_TRAIN_DIR)
        else:
            shutil.copy2(image_path, IMAGES_VAL_DIR)
            shutil.copy2(label_path, ANNOTATIONS_VAL_DIR)
    else:
        print(f"Image not found: {image_path}")
