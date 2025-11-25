"""
Convert large images into smaller ones.

If you get ZeroDivisionError, you may need to patch the sahi library, i.e. add these lines to slicing.py:137
if coco_annotation.area == 0:
    print("Skipping", coco_annotation)
    continue
"""

from shared_paths import *
from sahi.slicing import slice_coco

TILE_SIZE = 640
OVERLAP = 0.2

slice_coco(
    coco_annotation_file_path=ANNOTATIONS_COCO_FIXED_FILE,
    image_dir=IMAGES_DIR,
    output_dir=IMAGES_SLICED_DIR,
    output_coco_annotation_file_name=ANNOTATIONS_COCO_SLICED_FILE,
    slice_height=TILE_SIZE,
    slice_width=TILE_SIZE,
    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP
)
