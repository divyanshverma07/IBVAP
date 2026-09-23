import os
import cv2
os.environ["FLAGS_use_onednn"] = "0"
from ultralytics import YOLO
import easyocr
import threading
from queue import Queue, Empty

class ANPRModule:
    def __init__(self, model_path, confidence=0.25, ocr_interval=10):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ocr_interval = ocr_interval
        self.frame_count = 0
        self.last_ocr_results = []
        self.last_plates_info = []
        
        # Load EasyOCR
        self.ocr = easyocr.Reader(['en'], gpu=True) # It will fallback to CPU if GPU not available
        
        # Asynchronous OCR queue to maintain real-time FPS
        self.ocr_queue = Queue(maxsize=1)
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()

    def _ocr_worker(self):
        while True:
            try:
                plate_info, plate_bgr = self.ocr_queue.get()
                ocr_result = self.ocr.readtext(plate_bgr)
                if ocr_result:
                    best_res = max(ocr_result, key=lambda x: x[2])
                    plate_info["text"] = best_res[1]
                    plate_info["confidence"] = float(best_res[2])
                    print(f"[ANPR] Plate: {plate_info['text']} | Confidence: {plate_info['confidence']:.2f}")
                    self.last_ocr_results = [plate_info]
            except Exception as e:
                pass
            finally:
                self.ocr_queue.task_done()

    def process_frame(self, frame):
        self.frame_count += 1
        
        # Detect plates every 3rd frame to conserve CPU
        run_detection = (self.frame_count % 3 == 0) or not hasattr(self, 'last_plates_info')
        plates_info = getattr(self, 'last_plates_info', [])
        
        if run_detection:
            results = self.model(frame, conf=self.confidence, imgsz=320, verbose=False)
            plates_info = []

            if not results or results[0].boxes is None:
                self.last_plates_info = plates_info
                return plates_info, self.last_ocr_results

            run_ocr = (self.frame_count % self.ocr_interval == 0)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                plate_info = {
                    "bbox": (x1, y1, x2, y2),
                    "text": None,
                    "confidence": 0.0
                }

                if run_ocr and self.ocr_queue.empty():
                    plate = frame[y1:y2, x1:x2]
                    if plate.size > 0:
                        plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        plate_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        try:
                            self.ocr_queue.put_nowait((plate_info, plate_bgr))
                        except Exception:
                            pass
                
                plates_info.append(plate_info)
                
            self.last_plates_info = plates_info

        return plates_info, self.last_ocr_results
