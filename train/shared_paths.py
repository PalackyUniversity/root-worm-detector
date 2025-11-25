"""
Shared paths to make the run of individual scripts easier. Edit based on your needs - you may not need all of them.
"""

import os


WORKING_DIR = "."

IMAGES_DIR = os.path.join(WORKING_DIR, "data")
IMAGES_SLICED_DIR = os.path.join(WORKING_DIR, "data-sliced")

ANNOTATIONS_COCO_FILE = os.path.join(IMAGES_DIR, "output.json")
ANNOTATIONS_COCO_FIXED_FILE = os.path.join(IMAGES_DIR, "output_fixed.json")
__ANNOTATIONS_COCO_SLICED_FILENAME = "output_sliced.json"
ANNOTATIONS_COCO_SLICED_FILE = os.path.join(IMAGES_SLICED_DIR, f"{__ANNOTATIONS_COCO_SLICED_FILENAME}")
ANNOTATIONS_YOLO_DIR = os.path.join(IMAGES_SLICED_DIR, "coco2yolo")

FOLDER_LABELS = os.path.join(ANNOTATIONS_YOLO_DIR, f"labels/{__ANNOTATIONS_COCO_SLICED_FILENAME}_coco")
IMAGES_VAL_DIR = os.path.join(WORKING_DIR, "images/val")
IMAGES_TRAIN_DIR = os.path.join(WORKING_DIR, "images/train")
ANNOTATIONS_VAL_DIR = os.path.join(WORKING_DIR, "labels/val")
ANNOTATIONS_TRAIN_DIR = os.path.join(WORKING_DIR, "labels/train")
