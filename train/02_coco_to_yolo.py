from ultralytics.data.converter import convert_coco
from shared import *

convert_coco(
    labels_dir=IMAGES_SLICED_DIR,
    use_segments=True,
    use_keypoints=False,
    save_dir=ANNOTATIONS_YOLO_DIR
)
