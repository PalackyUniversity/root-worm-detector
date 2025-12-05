from sahi import AutoDetectionModel
from sahi.predict import predict
from shared import *

predict(
    model_type="ultralytics",
    model_path="../models/v2_best.pt",
    model_device="cuda:0",
    model_confidence_threshold=0.1,
    source=IMAGES_TEST_DIR,
    slice_height=TILE_SIZE,
    slice_width=TILE_SIZE,
    overlap_height_ratio=TILE_OVERLAP,
    overlap_width_ratio=TILE_OVERLAP,
    visual_text_size=1,
    visual_bbox_thickness=1,
    visual_text_thickness=1,
    export_pickle=True,
)
