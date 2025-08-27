# Root Worm Detector

Root Worm Detector is a Python-based application for detecting and visualizing root worms in images using deep learning. It provides an interactive GUI for batch processing, annotation, and inspection of predictions, making it suitable for research and practical applications in plant pathology and soil biology.

---

## Features

- **Drag & Drop File Addition:** Easily add images by dragging them into the application.
- **Interactive GUI:** Browse, annotate, and inspect predictions with a user-friendly interface.
- **Dynamic Visualization:** Bounding boxes, crosses, and prediction scores are dynamically scaled with zoom.
- **Selection Highlighting:** Prediction scores and contours change color when selected.
- **Batch Processing:** Process multiple images at once.
- **Custom Model Training:** Train your own detection model using provided scripts.
- **Export Results:** Save predictions and annotations for further analysis.

---

## Model Description

The detection backend is based on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework and [SAHI](https://github.com/obss/sahi) for efficient large image inference via slicing.

---

## Installation


1. **Install Python:**
```bash
   apt install python3 python3-pip python3-venv
   ```
2. **Install Git LFS:**
```bash
   git lfs install
   ```
3. **Clone the repository:**
   ```bash
   git clone https://github.com/PalackyUniversity/root-worm-detector.git
   cd root-worm-detector
   git lfs pull
   ```

4. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Usage

### Launching the GUI

```bash
python3 main.py
```

### Adding Images

- Use the "Import" buttons or **drag and drop** image files into the file list panel.

### Viewing Predictions

- Select an image and click "Predict" button to view predictions.
- Bounding boxes, crosses, and prediction scores are drawn and dynamically scaled.
- Click on predictions to select; selected predictions are highlighted and their scores change color. You can delete selected predictions if needed.
- You can zoom in/out using the +/- on the keyboard

### Exporting Results

- Annotations and predictions can be saved for further analysis.

---

## Training Your Own Model

You can train your own worm detection model using the scripts provided in the `train/` directory.

1. **Prepare your dataset** in the required format (images and corresponding masks/labels).
2. **Run the training pipeline:**
   - The training process is split into several scripts for preprocessing, tiling, conversion, and training.
   - Example workflow:
     ```bash
     # 1. Tile images and annotations
     python train/01_tiling.py

     # 2. Convert COCO annotations to YOLO format
     python train/02_coco_to_yolo.py

     # 3. Split dataset into train/val
     python train/03_split.py

     # 4. Train the model
     python train/04_train.py
     ```
   - Adjust script arguments and paths as needed for your dataset.

3. **Model Output:** The trained model weights will be saved and can be used by the GUI for inference.

---

## Project Structure

```
root-worm-detector/
├── logic/           # Core logic for image processing and prediction
├── train/           # Training and data preparation scripts
├── ui/              # GUI implementation
├── data/            # Example or user data (not included)
├── models/          # Pretrained or user-trained models
├── requirements.txt # Python dependencies
└── README.md
```
