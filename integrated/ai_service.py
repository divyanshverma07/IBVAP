import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import time
from datetime import datetime
import threading
import requests
import sqlite3

# --- MEMORY OPTIMIZATION FOR RENDER ---
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
# --------------------------------------

from flask import Flask, Response, jsonify, request
import numpy as np
from collections import deque
import config
from video_stream import VideoStream
from yolo_tracker import YOLOTracker
from fence_module import FenceModule
from anpr_module import ANPRModule
from drawer import Drawer

app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

WEB_SERVICE_URL = os.environ.get("WEB_SERVICE_URL", "http://localhost:5000")

def create_standby_frame(cam_label="CAM-01", status_msg="SURVEILLANCE STANDBY"):
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[:] = (18, 24, 38)
    cv2.rectangle(img, (20, 20), (940, 520), (45, 55, 75), 1)
    cv2.line(img, (480, 245), (480, 285), (70, 85, 115), 1)
    cv2.line(img, (460, 265), (500, 265), (70, 85, 115), 1)
    cv2.putText(img, f"[{cam_label}] {status_msg}", (280, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 175, 200), 2, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ret else None

class CameraPipeline:
    def __init__(self, cam_id, label):
        self.cam_id = cam_id
        self.label = label
        self.lock = threading.Lock()
        self.current_source = None
        self.active_name = None
        self.is_processing = False
        self.is_paused = False
        self.is_live_cam = False
        self.live_start_time = None
        self.live_duration_limit = 60
        self.restart_requested = False
        self.latest_frame_encoded = None
        self.live_telemetry_logs = deque(maxlen=30)
        self.target_logged_cooldown = {}
        self.state = {
            "cam_id": self.cam_id,
            "label": self.label,
            "stream_status": "STANDBY",
            "active_source": None,
            "is_live_cam": False,
            "live_time_remaining": 0,
            "active_tracks": 0,
            "incidents": 0,
            "fps": 0.0,
            "recent_alert": None,
            "error_message": None
        }

        self.custom_polygon = config.FENCE_POLYGON
        self.yolo_tracker = YOLOTracker(config.YOLO_MODEL_PATH, confidence=config.YOLO_CONFIDENCE)
        self.fence_module = FenceModule(self.custom_polygon, config.DB_NAME, config.EVIDENCE_DIR)
        
        # Monkey patch FenceModule to send to Web Service
        self.fence_module._log_to_db = self._remote_log_to_db
        self.fence_module._save_evidence = self._remote_save_evidence
        
        self.anpr_module = ANPRModule(config.ANPR_MODEL_PATH, confidence=config.ANPR_CONFIDENCE, ocr_interval=config.OCR_INTERVAL)
        self.drawer = Drawer(self.custom_polygon)

        self.standby_bytes = create_standby_frame(self.label)
        self.video = None
        self.tracked_vehicle_plates = {}
        self.anpr_logged_cooldown = {}

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _remote_log_to_db(self, person_id, label, confidence, centroid, status="VALID", evidence_image=None, incident_type="Perimeter Intrusion"):
        try:
            payload = {
                "person_id": person_id,
                "label": label,
                "confidence": confidence,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "status": status,
                "evidence_image": evidence_image,
                "incident_type": incident_type
            }
            requests.post(f"{WEB_SERVICE_URL}/api/internal/log_incident", json=payload, timeout=2)
        except Exception as e:
            print(f"[{self.label}] Failed to send log to web service: {e}")

    def _remote_save_evidence(self, frame, person_id, class_name="person", camera_label=""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_class = str(class_name).lower().replace(" ", "_")
        cam_prefix = f"{camera_label}_" if camera_label else ""
        filename = f"intrusion_{cam_prefix}{clean_class}_{person_id}_{timestamp}.jpg"
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            try:
                files = {'image': (filename, buf.tobytes(), 'image/jpeg')}
                resp = requests.post(f"{WEB_SERVICE_URL}/api/internal/upload_evidence", files=files, timeout=2)
                if resp.status_code == 200:
                    return resp.json().get("filepath", filename)
            except Exception as e:
                print(f"[{self.label}] Failed to send evidence to web service: {e}")
        return None

    def load_video(self, source_path, display_name):
        with self.lock:
            self.current_source = source_path
            self.active_name = display_name
            self.is_live_cam = False
            self.is_processing = True
            self.is_paused = False
            self.restart_requested = False
            self.state["stream_status"] = "LIVE"
            self.state["active_source"] = display_name
            self.state["is_live_cam"] = False
            self.state["live_time_remaining"] = 0
            self.state["error_message"] = None
            self.live_telemetry_logs.clear()
            self.target_logged_cooldown.clear()
            self.fence_module.person_states.clear()

    def start_live_camera(self, device_index=0):
        with self.lock:
            self.current_source = device_index
            self.active_name = f"Live Camera (Device {device_index})"
            self.is_live_cam = True
            self.live_start_time = time.time()
            self.is_processing = True
            self.is_paused = False
            self.restart_requested = False
            self.state["stream_status"] = "LIVE"
            self.state["active_source"] = self.active_name
            self.state["is_live_cam"] = True
            self.state["live_time_remaining"] = self.live_duration_limit
            self.state["error_message"] = None
            self.live_telemetry_logs.clear()
            self.target_logged_cooldown.clear()
            self.fence_module.person_states.clear()

    def pause(self):
        with self.lock:
            self.is_paused = True
            self.state["stream_status"] = "PAUSED"

    def resume(self):
        with self.lock:
            self.is_paused = False
            self.state["stream_status"] = "LIVE"

    def restart(self):
        with self.lock:
            self.restart_requested = True
            self.is_paused = False
            self.is_processing = True
            self.state["stream_status"] = "LIVE"

    def stop(self):
        with self.lock:
            self.is_processing = False
            self.is_paused = False
            self.is_live_cam = False
            self.current_source = None
            self.active_name = None
            self.latest_frame_encoded = None
            self.state["stream_status"] = "STANDBY"
            self.state["active_source"] = None
            self.state["active_tracks"] = 0
            self.state["recent_alert"] = None
            self.state["fps"] = 0.0
            self.state["is_live_cam"] = False
            self.state["live_time_remaining"] = 0
            self.state["error_message"] = None
            self.target_logged_cooldown.clear()
            self.fence_module.person_states.clear()

    def get_state(self):
        with self.lock:
            if self.is_live_cam and self.live_start_time:
                elapsed = time.time() - self.live_start_time
                rem = int(self.live_duration_limit - elapsed)
                if rem <= 0:
                    self.is_live_cam = False
                    self.is_processing = False
                    self.current_source = None
                    self.state["stream_status"] = "EXPIRED"
                    self.state["is_live_cam"] = False
                    self.state["live_time_remaining"] = 0
                    self.state["active_tracks"] = 0
                    self.state["recent_alert"] = None
                    self.state["fps"] = 0.0
                    self.latest_frame_encoded = None
                    if self.video:
                        self.video.release()
                        self.video = None
                else:
                    self.state["live_time_remaining"] = rem
            return self.state.copy()

    def generate_frames(self):
        while True:
            with self.lock:
                proc = self.is_processing
                frame = self.latest_frame_encoded
                status = self.state["stream_status"]
            
            if proc and frame is not None:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.033)
            else:
                if status == "EXPIRED":
                    exp_frame = create_standby_frame(self.label, "1-MIN LIVE LIMIT REACHED (STOPPED)")
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + exp_frame + b'\r\n')
                elif self.standby_bytes is not None:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + self.standby_bytes + b'\r\n')
                time.sleep(0.2)

    def _run_loop(self):
        loaded_source = None
        frame_counter = 0

        while True:
            with self.lock:
                proc = self.is_processing
                paused = self.is_paused
                target_src = self.current_source
                target_name = self.active_name
                is_live = self.is_live_cam
                start_live = self.live_start_time
                restart = self.restart_requested

            if is_live and start_live:
                elapsed = time.time() - start_live
                time_remaining = int(self.live_duration_limit - elapsed)
                if time_remaining <= 0:
                    if self.video:
                        self.video.release()
                        self.video = None
                    loaded_source = None
                    with self.lock:
                        self.is_processing = False
                        self.is_live_cam = False
                        self.current_source = None
                        self.state["live_time_remaining"] = 0
                        self.state["stream_status"] = "EXPIRED"
                        self.state["active_tracks"] = 0
                        self.state["recent_alert"] = None
                        self.state["fps"] = 0.0
                        self.latest_frame_encoded = None
                    time.sleep(0.1)
                    continue
                else:
                    with self.lock:
                        self.state["live_time_remaining"] = time_remaining

            if not proc or target_src is None:
                if self.video is not None:
                    self.video.release()
                    self.video = None
                    loaded_source = None
                    self.latest_frame_encoded = None
                    with self.lock:
                        if self.state["stream_status"] != "EXPIRED":
                            self.state["stream_status"] = "STANDBY"
                        self.state["active_tracks"] = 0
                        self.state["recent_alert"] = None
                        self.state["fps"] = 0.0
                time.sleep(0.1)
                continue

            if loaded_source != target_src or restart:
                with self.lock:
                    self.restart_requested = False
                if self.video is not None:
                    self.video.release()

                if target_src.startswith("http"):
                    # Download it locally first to avoid cv2.VideoCapture network hangs!
                    import urllib.request
                    local_dl_path = os.path.join(config.BASE_DIR, "integrated", "uploads", "downloaded_temp.mp4")
                    try:
                        print(f"[{self.label}] Downloading remote video: {target_src}")
                        # 5 second timeout for connection
                        req = urllib.request.Request(target_src, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=5) as response, open(local_dl_path, 'wb') as out_file:
                            out_file.write(response.read())
                        loaded_source = local_dl_path
                    except Exception as e:
                        print(f"[{self.label}] Failed to download video: {e}")
                        with self.lock:
                            self.is_processing = False
                            self.state["stream_status"] = "ERROR"
                            self.state["error_message"] = "Failed to download uploaded video."
                        self.video = None
                        loaded_source = None
                        time.sleep(0.5)
                        continue
                else:
                    loaded_source = target_src

                self.video = VideoStream(loaded_source, camera_id=self.label, max_dim=480) # REDUCED TO 480 FOR OPTIMIZATION
                if not self.video.connect():
                    with self.lock:
                        self.is_processing = False
                        self.state["stream_status"] = "ERROR"
                        self.state["error_message"] = f"Unable to open video source: {target_name}"
                    self.video = None
                    loaded_source = None
                    time.sleep(0.5)
                    continue

                self.tracked_vehicle_plates.clear()
                self.anpr_logged_cooldown.clear()
                with self.lock:
                    self.state["stream_status"] = "LIVE"
                    self.state["active_source"] = target_name
                frame_counter = 0

            if paused:
                with self.lock:
                    self.state["stream_status"] = "PAUSED"
                time.sleep(0.05)
                continue
            else:
                with self.lock:
                    if self.state["stream_status"] == "PAUSED":
                        self.state["stream_status"] = "LIVE"

            start_time = time.time()
            frame_counter += 1

            ret, frame = self.video.read_frame()
            if not ret:
                if not is_live and self.video and self.video.cap:
                    self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.fence_module.person_states.clear()
                    self.target_logged_cooldown.clear()
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            active_polygon = np.array([[int(w * 0.15), int(h * 0.18)], [int(w * 0.85), int(h * 0.18)], [int(w * 0.90), int(h * 0.90)], [int(w * 0.10), int(h * 0.90)]], dtype=np.int32)
            self.fence_module.fence_polygon = active_polygon
            self.drawer.fence_polygon = active_polygon

            yolo_detections = self.yolo_tracker.track(frame)
            anpr_boxes, last_ocr = self.anpr_module.process_frame(frame)

            for p_box in anpr_boxes:
                px1, py1, px2, py2 = p_box["bbox"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                p_text = p_box.get("text")
                p_conf = p_box.get("confidence", 0.0)
                if not p_text and last_ocr:
                    p_text = last_ocr[0].get("text")
                    p_conf = last_ocr[0].get("confidence", 0.0)

                for det in yolo_detections:
                    if det.get("class_name", "").lower() in ["car", "motorcycle", "bus", "truck"]:
                        vx1, vy1, vx2, vy2 = det["bbox"]
                        if (vx1 - 15) <= pcx <= (vx2 + 15) and (vy1 - 15) <= pcy <= (vy2 + 15):
                            vid = det["track_id"]
                            if vid != -1 and p_text:
                                self.tracked_vehicle_plates[vid] = {"plate": p_text, "confidence": p_conf, "last_seen": time.time()}
                            break

            fence_results = self.fence_module.process_detections(frame, yolo_detections, vehicle_plates=self.tracked_vehicle_plates, camera_label=self.label)

            now = time.time()
            for det in yolo_detections:
                tid = det["track_id"]
                if tid == -1: continue
                vclass = det.get("class_name", "obj").lower()
                conf = det.get("confidence", 0.0)

                last_logged = self.target_logged_cooldown.get(tid, 0)
                if (now - last_logged) > 12:
                    plate_info = self.tracked_vehicle_plates.get(tid)
                    has_plate = plate_info and plate_info.get("plate")
                    is_intruder = any(p["track_id"] == tid and p["status"] == "inside" for p in fence_results.get("person_results", []))

                    if is_live:
                        inc_type = "Live Camera Detection"
                        obj_label = f"[{self.label}] {vclass.upper()} #{tid} (Live Cam)"
                    elif has_plate:
                        inc_type = "ANPR Detection"
                        obj_label = f"[{self.label}] {vclass.upper()} #{tid} [{plate_info['plate']}]"
                        conf = plate_info.get("confidence", conf)
                    elif is_intruder:
                        inc_type = "Perimeter Intrusion"
                        obj_label = f"[{self.label}] {vclass.upper()} #{tid}"
                    else:
                        inc_type = "Surveillance Telemetry"
                        obj_label = f"[{self.label}] {vclass.upper()} #{tid}"

                    self.target_logged_cooldown[tid] = now
                    
                    vx1, vy1, vx2, vy2 = det["bbox"]
                    cx, cy = int((vx1 + vx2) / 2), int(vy2)
                    
                    # Generate Evidence and Log remotely
                    evidence_filename = self._remote_save_evidence(frame, tid, vclass, self.label)
                    self._remote_log_to_db(person_id=tid, label=obj_label, confidence=conf, centroid=(cx, cy), status="VALID", evidence_image=evidence_filename, incident_type=inc_type)

                    time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = {
                        "id": int(time.time() * 1000) % 10000000,
                        "time": time_display,
                        "camera": self.label,
                        "object": obj_label,
                        "confidence": round(conf * 100, 1),
                        "status": "VALID",
                        "raw_status": "VALID",
                        "evidence_url": f"/evidence/{evidence_filename}" if evidence_filename else None,
                        "evidence_filename": evidence_filename,
                        "incident": inc_type,
                        "is_live": True
                    }
                    self.live_telemetry_logs.appendleft(log_entry)

            proc_time = time.time() - start_time
            fps = 1.0 / proc_time if proc_time > 0 else 0.0

            annotated_frame = self.drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)
            cv2.putText(annotated_frame, self.label, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)

            if is_live:
                with self.lock:
                    rem_sec = self.state["live_time_remaining"]
                mins, secs = divmod(max(0, rem_sec), 60)
                timer_str = f"LIVE CAM LIMIT: {mins:02d}:{secs:02d} REMAINING"
                box_color = (0, 0, 220) if rem_sec <= 10 else (0, 140, 255)
                cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 370, 12), (annotated_frame.shape[1] - 20, 44), (0, 0, 0), -1)
                cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 370, 12), (annotated_frame.shape[1] - 20, 44), box_color, 2)
                cv2.putText(annotated_frame, timer_str, (annotated_frame.shape[1] - 355, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            active_intruders = [p for p in fence_results.get("person_results", []) if p["status"] == "inside"]
            with self.lock:
                self.state["active_tracks"] = len(yolo_detections)
                self.state["incidents"] = fence_results.get("outside_count", 0)
                self.state["fps"] = round(fps, 1)
                if active_intruders:
                    self.state["recent_alert"] = {
                        "track_id": active_intruders[0]["track_id"],
                        "class_name": f"[{self.label}] " + active_intruders[0].get("display_label", active_intruders[0].get("class_name", "OBJ")),
                        "confidence": active_intruders[0]["confidence"],
                        "camera_id": self.label
                    }
                else:
                    self.state["recent_alert"] = None

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60]) # REDUCED QUALITY FOR RENDER
            if ret:
                self.latest_frame_encoded = buffer.tobytes()

            # YIELD TO FLASK: Prevent CPU starvation on low-tier cloud instances
            # Without this, the background thread pegs the CPU and Gunicorn times out HTTP requests
            time.sleep(0.05)
            
cameras = {
    1: CameraPipeline(cam_id=1, label="CAM-01"),
    2: CameraPipeline(cam_id=2, label="CAM-02")
}

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    cam = cameras.get(cam_id, cameras[1])
    return Response(cam.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video_control', methods=['POST'])
def video_control():
    data = request.get_json() or {}
    action = data.get('action')
    cam_id = data.get('cam_id')
    
    target_cams = []
    if cam_id is not None and int(cam_id) in cameras:
        target_cams.append(cameras[int(cam_id)])
    else:
        target_cams = list(cameras.values())

    for cam in target_cams:
        if action == 'pause': cam.pause()
        elif action == 'resume': cam.resume()
        elif action == 'stop': cam.stop()
        elif action == 'restart': cam.restart()

    return jsonify({"success": True})

@app.route('/api/start_live_cam', methods=['POST'])
def start_live_cam():
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 1))
    device = int(data.get('device', 0))
    cam = cameras.get(cam_id, cameras[1])
    cam.start_live_camera(device)
    return jsonify({"success": True})

@app.route('/api/select_video', methods=['POST'])
def select_video():
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 1))
    source_url = data.get('source_url')
    filename = data.get('filename')
    
    if source_url and source_url.startswith("LOCAL_SAMPLE:"):
        source_url = os.path.join(config.BASE_DIR, source_url.split("LOCAL_SAMPLE:")[1])
    
    cam = cameras.get(cam_id, cameras[1])
    cam.load_video(source_url, filename)
    return jsonify({"success": True})

@app.route('/api/state')
def get_state():
    cam1_state = cameras[1].get_state()
    cam2_state = cameras[2].get_state()
    
    live_logs = list(cameras[1].live_telemetry_logs) + list(cameras[2].live_telemetry_logs)
    sorted_logs = sorted(live_logs, key=lambda x: x["time"], reverse=True)[:10]

    return jsonify({
        "cameras": {"1": cam1_state, "2": cam2_state},
        "live_logs": sorted_logs
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 AI INFERENCE SERVICE LIVE: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
