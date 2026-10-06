from pathlib import Path


class Model:
    CURRENT_MODEL_VERSION = 2
    MAX_OBJECT_SIZE = 1_000
    MIN_OBJECT_CONFIDENCE = 0.35
    NICE_THRESHOLD = 0.3
    PIPELINE_ID = "v8-s-mr4-seeded-unet-gb9-nice030-v2"
    MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
    DETECTOR_PATH = MODEL_DIR / "v8-s-mr4.pt"
    SHAPE_PATH = MODEL_DIR / "shape_unet_seeded.pt"
    CLASSIFIER_PATH = MODEL_DIR / "nice_gb9_v3.pkl"
    SHA256 = {
        "v8-s-mr4.pt": "c6c6190171937c6a739dc12033b2b66df4fa41c1fa753c481c48734f54c641b9",
        "shape_unet_seeded.pt": "df0fa4049758f9c8ef907e68bbf62b46d87f882a494f582be8c33a6bb5b6c5ca",
        "nice_gb9_v3.pkl": "7cfa11d55445a25a85f726623ca13d8a1f4fbc292c303be350373e4400cb3774",
    }

    DEFAULT_MODEL = "m"
    DETECTORS = {
        "s": {"filename": "v8-s-mr4.pt", "sha256": SHA256["v8-s-mr4.pt"],
              "confidence": .35, "pipeline": PIPELINE_ID},
        "m": {"filename": "v8-m-mr4.pt", "sha256": "bbb6bae542b07ede475e7fdaa2a4864c39edd9d25ba9486363503af27d668eed",
              "confidence": .35, "pipeline": "v8-m-mr4-seeded-unet-gb9-nice030-v2"},
    }
