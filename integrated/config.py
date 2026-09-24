import os
import numpy as np

# Dynamically resolve to SIH folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_PATH = os.environ.get("VIDEO_PATH", os.path.join(BASE_DIR, "test6.mp4"))
OUTPUT_VIDEO_PATH = os.environ.get("OUTPUT_VIDEO_PATH", os.path.join(BASE_DIR, "integrated", "output_integrated.mp4"))

# YOLO settings (Persons, Vehicles)
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "yolo11n.pt"))
YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.40"))

# ANPR settings (License Plates)
ANPR_MODEL_PATH = os.environ.get("ANPR_MODEL_PATH", os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"))
ANPR_CONFIDENCE = float(os.environ.get("ANPR_CONFIDENCE", "0.25"))
OCR_INTERVAL = int(os.environ.get("OCR_INTERVAL", "10"))

# Fence settings
DB_NAME = os.environ.get("DATABASE_URL", os.path.join(BASE_DIR, "integrated", "detections.db"))
EVIDENCE_DIR = os.environ.get("EVIDENCE_DIR", os.path.join(BASE_DIR, "integrated", "evidence"))

ENABLE_VIDEO_RECORDING = os.environ.get("ENABLE_VIDEO_RECORDING", "false").lower() == "true"

FENCE_POLYGON = np.array([
    [200, 150],
    [700, 150],
    [750, 500],
    [150, 500]
], dtype=np.int32)
