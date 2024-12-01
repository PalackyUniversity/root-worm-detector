from tqdm import tqdm
import random
import glob
import os

ENDING = ".png"
Y_FOLDER = "y-sliced"
X_FOLDER = "x-sliced"

for f in tqdm(glob.glob(f"{Y_FOLDER}/*{ENDING}")):
    folder = "train" if random.random() <= 0.7 else "valid"
    os.makedirs(f"data/{folder}/images", exist_ok=True)
    os.makedirs(f"data/{folder}/labels", exist_ok=True)
    os.system(f"cp {f} data/{folder}/labels/")
    os.system(f"cp {f.replace(Y_FOLDER, X_FOLDER)} data/{folder}/images/")
