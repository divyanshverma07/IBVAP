import cv2
import sqlite3
import os
from datetime import datetime

class FenceModule:
    def __init__(self, fence_polygon, db_name, evidence_dir):
        self.fence_polygon = fence_polygon
        self.db_name = db_name
        self.evidence_dir = evidence_dir
        
        os.makedirs(self.evidence_dir, exist_ok=True)
        # To ensure the directory containing the db exists
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        
        self.person_states = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trespass_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                person_id INTEGER NOT NULL,
                person_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                centroid_x INTEGER NOT NULL,
                centroid_y INTEGER NOT NULL,
                status TEXT NOT NULL,
                evidence_image TEXT,
                incident_type TEXT DEFAULT 'Perimeter Intrusion'
            )
        """)
        # Ensure incident_type column exists for backwards-compatibility with existing DBs
        cursor.execute("PRAGMA table_info(trespass_logs)")
        cols = [c[1] for c in cursor.fetchall()]
        if "incident_type" not in cols:
            cursor.execute("ALTER TABLE trespass_logs ADD COLUMN incident_type TEXT DEFAULT 'Perimeter Intrusion'")
            
        conn.commit()
        conn.close()

    def _log_to_db(self, person_id, label, confidence, centroid, status="VALID", evidence_image=None, incident_type="Perimeter Intrusion"):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO trespass_logs
            (timestamp, person_id, person_label, confidence, centroid_x, centroid_y, status, evidence_image, incident_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (current_time, int(person_id), str(label), float(confidence), 
             int(centroid[0]), int(centroid[1]), status, evidence_image, incident_type)
        )
        conn.commit()
        conn.close()

    def _save_evidence(self, frame, person_id, class_name="person", camera_label=""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_class = str(class_name).lower().replace(" ", "_")
        cam_prefix = f"{camera_label}_" if camera_label else ""
        filename = f"intrusion_{cam_prefix}{clean_class}_{person_id}_{timestamp}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)
        success = cv2.imwrite(filepath, frame)
        return filepath if success else None

    def process_detections(self, frame, detections, vehicle_plates=None, camera_label=""):
        if vehicle_plates is None:
            vehicle_plates = {}

        outside_count = 0
        inside_count = 0
        
        results = []

        for det in detections:
            track_id = det["track_id"]
            if track_id == -1:
                continue

            x1, y1, x2, y2 = det["bbox"]
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            foot_point = (foot_x, foot_y)

            distance = cv2.pointPolygonTest(self.fence_polygon, foot_point, False)
            is_inside = distance >= 0
            
            previous_state = self.person_states.get(track_id, None)

            class_name = det.get("class_name", "person").lower()
            plate_info = vehicle_plates.get(track_id)
            if plate_info and plate_info.get("plate"):
                display_label = f"{class_name.upper()} #{track_id} [{plate_info['plate']}]"
            elif class_name in ["car", "motorcycle", "bus", "truck"]:
                display_label = f"{class_name.upper()} #{track_id}"
            else:
                display_label = f"PERSON #{track_id}"

            if camera_label:
                display_label = f"[{camera_label}] {display_label}"

            if is_inside:
                outside_count += 1
                status = "inside"
                if previous_state != "inside":
                    print(f"[ALERT] {display_label} entered the RESTRICTED ZONE!")
                    evidence_path = self._save_evidence(frame, track_id, class_name, camera_label)
                    self._log_to_db(track_id, display_label, det["confidence"], foot_point, "VALID", evidence_path, "Perimeter Intrusion")
            else:
                inside_count += 1
                status = "outside"
                if previous_state == "inside":
                    print(f"[INFO] {display_label} left the restricted zone.")

            self.person_states[track_id] = status
            
            results.append({
                "track_id": track_id,
                "class_name": det["class_name"],
                "display_label": display_label,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "status": status,
                "foot_point": foot_point
            })

        return {
            "outside_count": outside_count,
            "inside_count": inside_count,
            "person_results": results
        }

