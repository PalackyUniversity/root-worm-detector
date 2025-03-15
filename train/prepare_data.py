from PIL import Image
from sahi.slicing import slice_image
from tqdm import tqdm
import cv2
import os

# Parameters for slicing
IMAGE_SIZE = 512
IMAGE_OVERLAP = 0.2

# Define paths
SOURCE_X_DIR = "x-original-4-weeks/"
SOURCE_Y_DIR = "y-original-4-weeks/"
SLICED_X_DIR = "x-sliced/"
SLICED_Y_DIR = "y-sliced/"

# Define formats
SOURCE_X_FORMAT = ".tif"
SOURCE_Y_FORMAT = ".tif"

# Create output directories if they don't exist
os.makedirs(SLICED_X_DIR, exist_ok=True)
os.makedirs(SLICED_Y_DIR, exist_ok=True)

# Loop through images and annotations
for annotation_filename in tqdm(os.listdir(SOURCE_X_DIR)):
    if annotation_filename.endswith(SOURCE_X_FORMAT):
        # Get corresponding annotation file
        image_filename = annotation_filename.replace(SOURCE_X_FORMAT, SOURCE_Y_FORMAT)

        # Paths for source and annotation
        image_path = os.path.join(SOURCE_X_DIR, image_filename)
        image = cv2.cvtColor(cv2.resize(cv2.imread(image_path), None, fx=1/2, fy=1/2), cv2.COLOR_BGR2RGB)
        annotation_path = os.path.join(SOURCE_Y_DIR, annotation_filename)

        # Slice source image
        slice_image(
            image=Image.fromarray(image),
            output_file_name=os.path.basename(image_path).split(".")[0],
            output_dir=SLICED_X_DIR,
            slice_height=IMAGE_SIZE,
            slice_width=IMAGE_SIZE,
            overlap_height_ratio=IMAGE_OVERLAP,
            overlap_width_ratio=IMAGE_OVERLAP
        )

        # Slice annotation
        # slice_image(
        #     image=annotation_path,
        #     output_file_name=os.path.basename(annotation_path).split(".")[0],
        #     output_dir=SLICED_Y_DIR,
        #     slice_height=IMAGE_SIZE,
        #     slice_width=IMAGE_SIZE,
        #     overlap_height_ratio=IMAGE_OVERLAP,
        #     overlap_width_ratio=IMAGE_OVERLAP
        # )
