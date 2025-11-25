from ultralytics import YOLO
import os

# Load a model pre-trained on COCO
model = YOLO("yolo11m-seg.pt")

# Train the model
results = model.train(
    data="dataset.yaml",
    epochs=400,
    imgsz=640,
    batch=24,  # set based on your VRAM
    workers=os.cpu_count(),
    pretrained=True,  # start from COCO weights
    patience=50,  # early stopping
    cache=True,  # cache dataset for speed
)

# Evaluate performance on validation set
metrics = model.val()
print(metrics)
