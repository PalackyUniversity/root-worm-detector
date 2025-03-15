from tqdm import tqdm
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"  # Set the segmentation_models framework
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Do not use GPU

import segmentation_models as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import cv2


# Data
DATA_X: str = "x-sliced"
DATA_Y: str = "y-sliced"
DATA_EMPTY_SKIP: float = 0.2
DATA_VAL_SIZE: float = 0.2
DATA_TEST_SIZE: float = 0.2

# Training
BATCH_SIZE: int = 8
VERBOSE: bool = True

# Epochs
TRAIN_LR_FACTOR = 0.4
TRAIN_LR_PATIENCE = 30
TRAIN_STOP_PATIENCE = 80
TRAIN_LENGTH = 200

TRAIN_METRICS: dict[str, str | sm.base.objects.Metric] = {
    "accuracy": "accuracy",
    "dice_loss": sm.losses.dice_loss,
    "precision": sm.metrics.precision,
    "recall": sm.metrics.recall,
    "f1-score": sm.metrics.f1_score,
    "f2-score": sm.metrics.f2_score
}
TRAIN_METRIC_MONITOR: str = "val_loss"

THRESHOLD_FOR_COUNTING = 0.1

MODEL_NAME: str = os.path.join("models", "model.keras")

# To make graphs prettyBATCH_SIZE: int = 8
sns.set_theme()

x_test = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in tqdm(glob.glob(f"{DATA_X}/*.png"))]
x_test = np.array(x_test, dtype="float32")

# Create model
model = sm.Unet("seresnet18", classes=2, encoder_weights=None, input_shape=x_test[0].shape)
model.compile("Adam", sm.losses.DiceLoss(), metrics=list(TRAIN_METRICS.values()))

# Load model if already trained
model.load_weights(MODEL_NAME)
y_pred = model.predict(x_test, batch_size=BATCH_SIZE)

os.makedirs("output", exist_ok=True)

for n in tqdm(range(len(y_pred))):
    _, ax = plt.subplots(1, 3, figsize=(19, 12))

    for a in ax:
        a.grid(None)
        a.axis("off")

    # Count objects in the test and prediction
    contours_pred, _ = cv2.findContours((y_pred[n, :, :, 1] >= THRESHOLD_FOR_COUNTING).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ax[0].imshow(x_test[n, :, :].astype("uint8"))
    ax[1].imshow(y_pred[n, :, :, 1], cmap="gray")
    ax[2].imshow(np.log(y_pred[n, :, :, 1]), cmap="gray")

    ax[0].set_title("Original")
    ax[1].set_title(f"AI (count: {len(contours_pred)})")
    ax[2].set_title("AI - prob")

    plt.savefig(f"output/{n}.jpg")