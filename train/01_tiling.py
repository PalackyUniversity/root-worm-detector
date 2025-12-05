"""
Convert large images into smaller ones.

If you get ZeroDivisionError, you may need to patch the sahi library, i.e. add these lines to slicing.py:137
if coco_annotation.area == 0:
    print("Skipping", coco_annotation)
    continue
"""

from sahi.slicing import slice_coco
from shared import *

slice_coco(
    coco_annotation_file_path=ANNOTATIONS_COCO_FIXED_FILE,
    image_dir=IMAGES_DIR,
    output_dir=IMAGES_SLICED_DIR,
    output_coco_annotation_file_name=ANNOTATIONS_COCO_SLICED_FILE,
    slice_height=TILE_SIZE,
    slice_width=TILE_SIZE,
    overlap_height_ratio=TILE_OVERLAP,
    overlap_width_ratio=TILE_OVERLAP
)
