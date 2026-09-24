import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    writer = None
    if config.ENABLE_VIDEO_RECORDING:
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

        yolo_detections = yolo_tracker.track(frame)
        fence_results = fence_module.process_detections(frame, yolo_detections)
        anpr_boxes, last_ocr = anpr_module.process_frame(frame)

        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        frame = drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)

        if writer:
            writer.write(frame)
            
        cv2.imshow("Integrated Surveillance Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

def run_dashboard():
    print("Starting Web Dashboard...")
    from app import app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--desktop":
        main()
    else:
        run_dashboard()
