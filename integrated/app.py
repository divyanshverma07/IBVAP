import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import time
from datetime import datetime
import sqlite3
import threading

# --- MEMORY OPTIMIZATION FOR RENDER ---
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
# --------------------------------------

from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from werkzeug.utils import secure_filename
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

# Upload configuration (allow up to 500 MB for surveillance videos)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_standby_frame(cam_label="CAM-01", status_msg="SURVEILLANCE STANDBY"):
    """Generate a sleek dark surveillance standby frame for MJPEG stream when idle."""
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[:] = (18, 24, 38)
    # HUD borders & crosshairs
    cv2.rectangle(img, (20, 20), (940, 520), (45, 55, 75), 1)
    cv2.line(img, (480, 245), (480, 285), (70, 85, 115), 1)
    cv2.line(img, (460, 265), (500, 265), (70, 85, 115), 1)
    
    cv2.putText(img, f"[{cam_label}] {status_msg}", (280, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 175, 200), 2, cv2.LINE_AA)
    cv2.putText(img, "Select Video or Start Live Camera Session (1-Min Limit)", (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 140), 1, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ret else None

class CameraPipeline:
    """Independent multi-threaded AI pipeline for each camera (CAM-01 / CAM-02)."""
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
        self.live_duration_limit = 60  # Strict 1-minute limit for live camera
        self.restart_requested = False
        self.latest_frame_encoded = None
        
        self.live_telemetry_logs = deque(maxlen=30)
        self.target_logged_cooldown = {}
        
        self.state = {
            "cam_id": self.cam_id,
            "label": self.label,
            "stream_status": "STANDBY",  # "STANDBY" | "LIVE" | "PAUSED" | "EXPIRED" | "ERROR"
            "active_source": None,
            "is_live_cam": False,
            "live_time_remaining": 0,
            "active_tracks": 0,
            "incidents": 0,
            "fps": 0.0,
            "recent_alert": None,
            "error_message": None
        }

        # Initialize dedicated AI models for this camera
        self.custom_polygon = config.FENCE_POLYGON
        self.yolo_tracker = YOLOTracker(config.YOLO_MODEL_PATH, confidence=config.YOLO_CONFIDENCE)
        self.fence_module = FenceModule(self.custom_polygon, config.DB_NAME, config.EVIDENCE_DIR)
        self.anpr_module = ANPRModule(config.ANPR_MODEL_PATH, confidence=config.ANPR_CONFIDENCE, ocr_interval=config.OCR_INTERVAL)
        self.drawer = Drawer(self.custom_polygon)

        self.standby_bytes = create_standby_frame(self.label)
        self.video = None
        self.tracked_vehicle_plates = {}
        self.anpr_logged_cooldown = {}

        # Launch background processing worker
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

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
        """Start physical camera/webcam with strict 60-second limit."""
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
            print(f"[{self.label}] Started Live Camera with 60s prototype limit.")

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
            # Enforce 1-minute live camera limit check on state query
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
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.033)  # ~30 FPS
            else:
                # If expired, render expired frame; else standby
                if status == "EXPIRED":
                    exp_frame = create_standby_frame(self.label, "1-MIN LIVE LIMIT REACHED (STOPPED)")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + exp_frame + b'\r\n')
                elif self.standby_bytes is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + self.standby_bytes + b'\r\n')
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

            # 1-Minute Live Camera Limit Check (Run immediately at top of loop)
            if is_live and start_live:
                elapsed = time.time() - start_live
                time_remaining = int(self.live_duration_limit - elapsed)
                
                if time_remaining <= 0:
                    print(f"[{self.label}] ⏱️ 1-Minute limit reached for live camera. Stopping.")
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

            # Standby / Stopped condition
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

            # Switch source or restart
            if loaded_source != target_src or restart:
                with self.lock:
                    self.restart_requested = False
                if self.video is not None:
                    self.video.release()

                loaded_source = target_src
                self.video = VideoStream(loaded_source, camera_id=self.label, max_dim=720)
                if not self.video.connect():
                    print(f"[{self.label}] ERROR: Failed to connect to {loaded_source}")
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
                    # Loop video file continuously and reset track states for ongoing logging
                    self.video.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.fence_module.person_states.clear()
                    self.target_logged_cooldown.clear()
                time.sleep(0.01)
                continue

            # 1. Adaptive Fence Polygon for this frame resolution
            h, w = frame.shape[:2]
            active_polygon = np.array([
                [int(w * 0.15), int(h * 0.18)],
                [int(w * 0.85), int(h * 0.18)],
                [int(w * 0.90), int(h * 0.90)],
                [int(w * 0.10), int(h * 0.90)]
            ], dtype=np.int32)
            self.fence_module.fence_polygon = active_polygon
            self.drawer.fence_polygon = active_polygon

            # 2. YOLO Tracker
            yolo_detections = self.yolo_tracker.track(frame)

            # 3. ANPR Logic
            anpr_boxes, last_ocr = self.anpr_module.process_frame(frame)

            # 4. Associate plates with vehicles
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
                                self.tracked_vehicle_plates[vid] = {
                                    "plate": p_text,
                                    "confidence": p_conf,
                                    "last_seen": time.time()
                                }
                            break

            # 5. Fence Logic with camera label tagging
            fence_results = self.fence_module.process_detections(
                frame, yolo_detections, vehicle_plates=self.tracked_vehicle_plates, camera_label=self.label
            )

            # 6. Real-Time Telemetry & Target Logging (Live Webcam & Active Video Tracks)
            now = time.time()
            for det in yolo_detections:
                tid = det["track_id"]
                if tid == -1:
                    continue
                vclass = det.get("class_name", "obj").lower()
                conf = det.get("confidence", 0.0)

                last_logged = self.target_logged_cooldown.get(tid, 0)
                # Log on first appearance, then every 12 seconds per active track
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
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    snap_filename = f"{self.label}_{vclass}_{tid}_{timestamp_str}.jpg"
                    snap_path = os.path.join(config.EVIDENCE_DIR, snap_filename)
                    cv2.imwrite(snap_path, frame)

                    vx1, vy1, vx2, vy2 = det["bbox"]
                    cx, cy = int((vx1 + vx2) / 2), int(vy2)

                    try:
                        self.fence_module._log_to_db(
                            person_id=tid,
                            label=obj_label,
                            confidence=conf,
                            centroid=(cx, cy),
                            status="VALID",
                            evidence_image=snap_path,
                            incident_type=inc_type
                        )
                    except Exception as e:
                        print(f"[{self.label}] DB log error: {e}")

                    log_entry = {
                        "id": int(time.time() * 1000) % 10000000,
                        "time": time_display,
                        "camera": self.label,
                        "object": obj_label,
                        "confidence": round(conf * 100, 1),
                        "status": "VALID",
                        "raw_status": "VALID",
                        "evidence_url": f"/evidence/{snap_filename}",
                        "evidence_filename": snap_filename,
                        "incident": inc_type,
                        "is_live": True
                    }
                    self.live_telemetry_logs.appendleft(log_entry)
                    print(f"[{self.label}] 🟢 LIVE LOG: {inc_type} - {obj_label}")

            # 6. Calculate FPS
            proc_time = time.time() - start_time
            fps = 1.0 / proc_time if proc_time > 0 else 0.0

            # 7. Draw HUD Overlays
            annotated_frame = self.drawer.draw(frame, yolo_detections, fence_results, anpr_boxes, last_ocr, fps)

            # Camera ID Watermark on HUD
            cv2.putText(annotated_frame, self.label, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)

            # If Live Camera: Overlay visual 1-minute countdown timer directly onto frame
            if is_live:
                with self.lock:
                    rem_sec = self.state["live_time_remaining"]
                mins, secs = divmod(max(0, rem_sec), 60)
                timer_str = f"LIVE CAM LIMIT: {mins:02d}:{secs:02d} REMAINING"
                # Draw red/amber countdown box
                box_color = (0, 0, 220) if rem_sec <= 10 else (0, 140, 255)
                cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 370, 12), (annotated_frame.shape[1] - 20, 44), (0, 0, 0), -1)
                cv2.rectangle(annotated_frame, (annotated_frame.shape[1] - 370, 12), (annotated_frame.shape[1] - 20, 44), box_color, 2)
                cv2.putText(annotated_frame, timer_str, (annotated_frame.shape[1] - 355, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # 8. Update Camera State
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

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                self.latest_frame_encoded = buffer.tobytes()

# Initialize Dual Camera Pipelines
cameras = {
    1: CameraPipeline(cam_id=1, label="CAM-01"),
    2: CameraPipeline(cam_id=2, label="CAM-02")
}

def processing_thread():
    """Heartbeat coordinator keeping backward compatibility with run_live.py."""
    print("[SYSTEM] Dual-Camera Surveillance Pipelines initialized (CAM-01 & CAM-02).")
    while True:
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id=1):
    cam = cameras.get(cam_id, cameras[1])
    return Response(cam.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evidence/<path:filename>')
def get_evidence(filename):
    return send_from_directory(config.EVIDENCE_DIR, filename)

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload custom video and assign to Camera 1 or Camera 2."""
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "Empty filename selected"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            "success": False, 
            "error": f"Invalid file format. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    cam_id = int(request.form.get('cam_id', 1))
    cam = cameras.get(cam_id, cameras[1])

    original_name = file.filename
    clean_name = secure_filename(original_name) or f"upload_{int(time.time())}.mp4"
    unique_filename = f"{int(time.time())}_{clean_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save video: {str(e)}"}), 500

    cam.load_video(save_path, original_name)

    return jsonify({
        "success": True, 
        "cam_id": cam_id,
        "filename": original_name,
        "message": f"Assigned {original_name} to {cam.label}. Processing started!"
    })

@app.route('/api/select_video', methods=['POST'])
def select_video():
    """Select a sample clip or previously uploaded video for Camera 1 or Camera 2."""
    data = request.get_json() or {}
    filename = data.get('filename')
    source_type = data.get('type', 'sample')
    cam_id = int(data.get('cam_id', 1))
    cam = cameras.get(cam_id, cameras[1])

    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400

    # Live camera shortcut
    if filename == 'live' or filename == '0':
        cam.start_live_camera(0)
        return jsonify({
            "success": True,
            "cam_id": cam_id,
            "filename": "Live Camera (Device 0)",
            "message": f"Started Live Camera on {cam.label} with 1-minute limit!"
        })

    safe_name = os.path.basename(filename)
    target_path = None

    if source_type == 'uploaded':
        candidate = os.path.join(UPLOAD_DIR, safe_name)
        if os.path.exists(candidate):
            target_path = candidate
    else:
        candidate = os.path.join(config.BASE_DIR, safe_name)
        if os.path.exists(candidate):
            target_path = candidate
        else:
            candidate_up = os.path.join(UPLOAD_DIR, safe_name)
            if os.path.exists(candidate_up):
                target_path = candidate_up

    if not target_path or not os.path.exists(target_path):
        return jsonify({"success": False, "error": f"Video file '{safe_name}' not found."}), 404

    cam.load_video(target_path, safe_name)

    return jsonify({
        "success": True, 
        "cam_id": cam_id,
        "filename": safe_name, 
        "message": f"Assigned {safe_name} to {cam.label}."
    })

@app.route('/api/start_live_cam', methods=['POST'])
def start_live_cam():
    """Start Live Camera session on specified camera with strict 1-minute limit."""
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 1))
    device = int(data.get('device', 0))
    cam = cameras.get(cam_id, cameras[1])

    cam.start_live_camera(device)
    return jsonify({
        "success": True,
        "cam_id": cam_id,
        "device": device,
        "message": f"Live Camera started on {cam.label}. 60-second limit is active!"
    })

@app.route('/api/video_control', methods=['POST'])
def video_control():
    """Handle pause, resume, stop, restart for Camera 1, Camera 2, or both."""
    data = request.get_json() or {}
    action = data.get('action')
    cam_id = data.get('cam_id')

    target_cams = []
    if cam_id is not None and int(cam_id) in cameras:
        target_cams.append(cameras[int(cam_id)])
    else:
        # Default or 'all': control both cameras
        target_cams = list(cameras.values())

    for cam in target_cams:
        if action == 'pause':
            cam.pause()
        elif action == 'resume':
            cam.resume()
        elif action == 'stop':
            cam.stop()
        elif action == 'restart':
            cam.restart()

    return jsonify({"success": True, "action": action, "cam_id": cam_id})

@app.route('/api/available_videos')
def available_videos():
    """Return list of sample videos and uploaded files."""
    def get_file_size_str(bytes_size):
        if bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        return f"{bytes_size / (1024 * 1024):.1f} MB"

    samples = []
    if os.path.exists(config.BASE_DIR):
        for fname in sorted(os.listdir(config.BASE_DIR)):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                fpath = os.path.join(config.BASE_DIR, fname)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    samples.append({
                        "filename": fname,
                        "size": size,
                        "size_str": get_file_size_str(size),
                        "type": "sample"
                    })

    uploads = []
    if os.path.exists(UPLOAD_DIR):
        for fname in sorted(os.listdir(UPLOAD_DIR), reverse=True):
            if allowed_file(fname):
                fpath = os.path.join(UPLOAD_DIR, fname)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    display_name = fname
                    if "_" in fname and fname.split("_")[0].isdigit():
                        display_name = fname.split("_", 1)[1]
                    uploads.append({
                        "filename": fname,
                        "display_name": display_name,
                        "size": size,
                        "size_str": get_file_size_str(size),
                        "type": "uploaded"
                    })

    return jsonify({
        "samples": samples,
        "uploads": uploads,
        "cam1": cameras[1].get_state(),
        "cam2": cameras[2].get_state()
    })

@app.route('/api/incidents/<int:incident_id>/status', methods=['POST'])
def update_incident_status(incident_id):
    data = request.get_json() or {}
    new_status = data.get('status', 'FALSE_ALARM')
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        cursor.execute("UPDATE trespass_logs SET status = ? WHERE id = ?", (new_status, incident_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": incident_id, "status": new_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/state')
def get_state():
    """Return unified state for both cameras + incident logs."""
    db_logs = []
    total_incidents = 0
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM trespass_logs")
        total_incidents = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA table_info(trespass_logs)")
        cols = [c[1] for c in cursor.fetchall()]
        has_incident_type = "incident_type" in cols

        query = (
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, incident_type "
            "FROM trespass_logs ORDER BY id DESC LIMIT 25"
            if has_incident_type else
            "SELECT id, timestamp, person_label, confidence, status, evidence_image, 'Perimeter Intrusion' "
            "FROM trespass_logs ORDER BY id DESC LIMIT 25"
        )
        cursor.execute(query)
        for row in cursor.fetchall():
            ev_path = row[5]
            ev_filename = os.path.basename(ev_path) if ev_path else None
            ev_url = f"/evidence/{ev_filename}" if ev_filename else None

            raw_status = str(row[4]).upper() if row[4] else "VALID"
            ui_status = "FALSE_ALARM" if "FALSE" in raw_status else "VALID"

            label_str = str(row[2])
            cam_tag = "CAM-01"
            if "[CAM-02]" in label_str:
                cam_tag = "CAM-02"
            elif "[CAM-01]" in label_str:
                cam_tag = "CAM-01"

            db_logs.append({
                "id": row[0],
                "time": row[1],
                "camera": cam_tag,
                "object": label_str,
                "confidence": round(row[3] * 100, 1) if row[3] else 0,
                "status": ui_status,
                "raw_status": row[4],
                "evidence_url": ev_url,
                "evidence_filename": ev_filename,
                "incident": row[6] if row[6] else "Perimeter Intrusion",
                "is_live": False
            })
        conn.close()
    except Exception as e:
        total_incidents = 0
        db_logs = []

    cam1_state = cameras[1].get_state()
    cam2_state = cameras[2].get_state()

    # Collect in-memory live logs from active camera pipelines
    live_logs = list(cameras[1].live_telemetry_logs) + list(cameras[2].live_telemetry_logs)

    # Merge live logs with DB logs, keeping latest first, deduplicating
    seen_keys = set()
    combined_logs = []
    
    # Fresh live logs first
    for item in sorted(live_logs, key=lambda x: x["time"], reverse=True):
        key = (item.get("evidence_filename") or item.get("object"), item.get("time"))
        if key not in seen_keys:
            seen_keys.add(key)
            combined_logs.append(item)

    # Then DB logs
    for item in db_logs:
        key = (item.get("evidence_filename") or item.get("object"), item.get("time"))
        if key not in seen_keys:
            seen_keys.add(key)
            combined_logs.append(item)

    combined_logs = combined_logs[:25]

    # Active alert across both cameras
    active_alert = cam1_state.get("recent_alert") or cam2_state.get("recent_alert")

    return jsonify({
        "cameras": {
            "1": cam1_state,
            "2": cam2_state
        },
        "stream_status": cam1_state["stream_status"],
        "active_video": cam1_state["active_source"],
        "is_processing": cam1_state["stream_status"] == "LIVE" or cam2_state["stream_status"] == "LIVE",
        "fps": cam1_state["fps"],
        "active_tracks": cam1_state["active_tracks"] + cam2_state["active_tracks"],
        "total_incidents": total_incidents,
        "recent_alert": active_alert,
        "system_health": "ONLINE",
        "model": "yolo11n.pt",
        "logs": combined_logs
    })

if __name__ == '__main__':
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print(f"🚀 DUAL-CAMERA BORDER SURVEILLANCE DASHBOARD LIVE: http://0.0.0.0:{port}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
