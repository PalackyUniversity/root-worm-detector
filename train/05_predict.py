from sahi import AutoDetectionModel

# detection_model = AutoDetectionModel.from_pretrained(
#     model_type="ultralytics",
#     model_path="/media/tadeas/Data4/root-worm-detector-repa/runs/segment/train4/weights/best.pt",
#     confidence_threshold=0.6,
#     device="cpu",  # or 'cuda:0'
# )
#
# from sahi.predict import get_sliced_prediction
#
# result = get_sliced_prediction(
#     "demo_data/small-vehicles1.jpeg",
#     detection_model,
#     slice_height=640,
#     slice_width=640,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )


from sahi.predict import predict

TILE_SIZE = 640
OVERLAP = 0.2

predict(
    model_type="ultralytics",
    model_path="../models/best.pt",
    model_device="cuda:0",  # or 'cuda:0'
    model_confidence_threshold=0.1,
    source="data",
    slice_height=TILE_SIZE,
    slice_width=TILE_SIZE,
    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP,
    visual_text_size=1,
    visual_bbox_thickness=1,
    visual_text_thickness=1,
    export_pickle=True,
)