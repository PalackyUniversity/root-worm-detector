from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")

# Train the model
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)