import os
import numpy as np

# Dynamically resolve to SIH folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_PATH = os.path.join(BASE_DIR, "test6.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "integrated", "output_integrated.mp4")

# YOLO settings (Persons, Vehicles)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
YOLO_CONFIDENCE = 0.40

# ANPR settings (License Plates)
ANPR_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
ANPR_CONFIDENCE = 0.25
OCR_INTERVAL = 10

# Fence settings
DB_NAME = os.path.join(BASE_DIR, "integrated", "detections.db")
EVIDENCE_DIR = os.path.join(BASE_DIR, "integrated", "evidence")

FENCE_POLYGON = np.array([
    [200, 150],
    [700, 150],
    [750, 500],
    [150, 500]
], dtype=np.int32)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app import app, processing_thread
    import threading
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    print("\n" + "="*50)
    print("🚀 BORDER SURVEILLANCE DASHBOARD LIVE: http://127.0.0.1:5000")
    print(" Open your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

