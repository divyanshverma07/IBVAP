import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import cv2

import config
from video_stream import VideoStream
from yolo_tracker import YOLOTracker
from fence_module import FenceModule
from anpr_module import ANPRModule
from drawer import Drawer

from zone_selector import select_zone

def main():
    # Initialize Video
    video = VideoStream(config.VIDEO_PATH, camera_id="MAIN-CAM")
    if not video.connect():
        return

    # Grab the first frame for the user to draw the polygon
    ret, first_frame = video.read_frame()
    if not ret:
        print("[ERROR] Could not read the first frame.")
        return
        
    print("\n========================================")
    print(" INTERACTIVE ZONE SELECTION")
    print("========================================")
    print(" Please draw the Restricted Zone on the pop-up window.")
    print(" Click to add points, and press ENTER when done.")
    
    custom_polygon = select_zone(first_frame)
    print(" Zone successfully set!")

    # Initialize Modules with the custom polygon
    yolo_tracker = YOLOTracker(config.YOLO_MODEL_PATH, confidence=config.YOLO_CONFIDENCE)
    fence_module = FenceModule(custom_polygon, config.DB_NAME, config.EVIDENCE_DIR)
    anpr_module = ANPRModule(config.ANPR_MODEL_PATH, confidence=config.ANPR_CONFIDENCE, ocr_interval=config.OCR_INTERVAL)
    drawer = Drawer(custom_polygon)

    # Video Writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, video.fps, (video.width, video.height))

    print("========================================")
    print(" INTEGRATED SURVEILLANCE PIPELINE STARTED")
    print("========================================")
    print(" Press 'Q' to exit.")

    while True:
        start_time = time.time()

        ret, frame = video.read_frame()
        if not ret:
            print("Video ended.")
            break

        # 1. Run Unified YOLO Tracker (Persons, Vehicles, etc.)
        yolo_detections = yolo_tracker.track(frame)

        # 2. Run Fence Logic (Filters for Persons internally)
        fence_results = fence_module.process_detections(frame, yolo_detections)

        # 3. Run ANPR Logic (Plates + OCR)
        anpr_boxes, last_ocr = anpr_module.process_frame(frame)

        # 4. Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        # 5. Draw everything onto the frame
        frame = drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)

        # 6. Show and Write Frame
        writer.write(frame)
        cv2.imshow("Integrated Surveillance Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video.release()
    writer.release()
    cv2.destroyAllWindows()

def run_dashboard():
    from app import app, processing_thread
    import threading
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    print("\n" + "="*50)
    print("🚀 BORDER SURVEILLANCE DASHBOARD LIVE: http://127.0.0.1:5000")
    print(" Open your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--desktop":
        main()
    else:
        run_dashboard()

