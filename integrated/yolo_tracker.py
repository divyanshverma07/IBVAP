from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path, confidence=0.40):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def track(self, frame):
        # We track multiple classes: 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=320,
            verbose=False
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            track_id = int(box.id[0]) if box.id is not None else -1

            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })

        return detections
