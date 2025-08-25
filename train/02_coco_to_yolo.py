from ultralytics.data.converter import convert_coco
from pathlib import Path

convert_coco(
    labels_dir=Path("sliced"),
    use_segments=True,
    use_keypoints=False,
    save_dir=Path("coco_to_yolo_output")
)
