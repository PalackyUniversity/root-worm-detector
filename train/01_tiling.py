from sahi.slicing import slice_coco

TILE_SIZE = 640
OVERLAP = 0.2

slice_coco(
    coco_annotation_file_path="output.json",
    image_dir="data",
    output_dir="sliced/",
    output_coco_annotation_file_name="sliced_output.json",
    slice_height=TILE_SIZE,
    slice_width=TILE_SIZE,
    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP
)
