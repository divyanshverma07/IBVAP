import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from datetime import datetime
import sqlite3
import requests
import json

from flask import Flask, render_template, Response, jsonify, send_from_directory, request, stream_with_context
from werkzeug.utils import secure_filename
import config

app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv'}

AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:5001")
PUBLIC_WEB_URL = os.environ.get("PUBLIC_WEB_URL", "http://localhost:5000") # Used by AI service to download uploads

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _init_db():
    conn = sqlite3.connect(config.DB_NAME)
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
    cursor.execute("PRAGMA table_info(trespass_logs)")
    cols = [c[1] for c in cursor.fetchall()]
    if "incident_type" not in cols:
        cursor.execute("ALTER TABLE trespass_logs ADD COLUMN incident_type TEXT DEFAULT 'Perimeter Intrusion'")
    conn.commit()
    conn.close()

_init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id=1):
    # Proxy the video feed from the AI Service
    try:
        req = requests.get(f"{AI_SERVICE_URL}/video_feed/{cam_id}", stream=True, timeout=5)
        return Response(stream_with_context(req.iter_content(chunk_size=1024)), content_type=req.headers['content-type'])
    except Exception as e:
        print(f"Error proxying video feed: {e}")
        return "AI Service Unavailable", 503

@app.route('/evidence/<path:filename>')
def get_evidence(filename):
    return send_from_directory(config.EVIDENCE_DIR, filename)

@app.route('/uploads/<path:filename>')
def get_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "Empty filename selected"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file format"}), 400

    cam_id = int(request.form.get('cam_id', 1))
    clean_name = secure_filename(file.filename) or f"upload_{int(time.time())}.mp4"
    unique_filename = f"{int(time.time())}_{clean_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save video: {str(e)}"}), 500

    # Tell AI Service to load it from the public URL
    source_url = f"{PUBLIC_WEB_URL}/uploads/{unique_filename}"
    try:
        resp = requests.post(f"{AI_SERVICE_URL}/api/select_video", json={
            "cam_id": cam_id,
            "source_url": source_url,
            "filename": file.filename
        }, timeout=5)
        if resp.status_code == 200:
            return jsonify({
                "success": True, 
                "cam_id": cam_id,
                "filename": file.filename,
                "message": f"Assigned {file.filename} to CAM-{cam_id:02d}. Processing started!"
            })
    except Exception as e:
        print(f"Failed to communicate with AI Service: {e}")

    return jsonify({"success": False, "error": "AI Service unavailable"}), 500

@app.route('/api/select_video', methods=['POST'])
def select_video():
    data = request.get_json() or {}
    filename = data.get('filename')
    source_type = data.get('type', 'sample')
    cam_id = int(data.get('cam_id', 1))

    if not filename:
        return jsonify({"success": False, "error": "Filename required"}), 400

    if filename == 'live' or filename == '0':
        try:
            requests.post(f"{AI_SERVICE_URL}/api/start_live_cam", json={"cam_id": cam_id, "device": 0}, timeout=2)
            return jsonify({
                "success": True,
                "cam_id": cam_id,
                "filename": "Live Camera (Device 0)",
                "message": f"Started Live Camera on CAM-{cam_id:02d}!"
            })
        except Exception as e:
            return jsonify({"success": False, "error": "AI Service unavailable"}), 500

    safe_name = os.path.basename(filename)
    
    # Check if local sample or upload
    # AI service won't have local files if separated, so we serve them
    if source_type == 'uploaded':
        source_url = f"{PUBLIC_WEB_URL}/uploads/{safe_name}"
    else:
        # It's a sample file. If it's not in uploads, maybe we can't serve it directly if it's in root.
        # But for prototype, let's assume sample files are accessible or we pass the local absolute path
        # Assuming AI service might be on the same disk if using Render persistent disk.
        # If not, passing absolute path will fail. We'll pass the URL to our root files if we add an endpoint.
        # Let's add a quick root endpoint for samples.
        source_url = f"{PUBLIC_WEB_URL}/samples/{safe_name}"

    try:
        requests.post(f"{AI_SERVICE_URL}/api/select_video", json={
            "cam_id": cam_id,
            "source_url": source_url,
            "filename": safe_name
        }, timeout=5)
        return jsonify({
            "success": True, 
            "cam_id": cam_id,
            "filename": safe_name, 
            "message": f"Assigned {safe_name} to CAM-{cam_id:02d}."
        })
    except Exception as e:
        return jsonify({"success": False, "error": "AI Service unavailable"}), 500

@app.route('/samples/<path:filename>')
def get_sample(filename):
    return send_from_directory(config.BASE_DIR, filename)

@app.route('/api/start_live_cam', methods=['POST'])
def start_live_cam():
    try:
        resp = requests.post(f"{AI_SERVICE_URL}/api/start_live_cam", json=request.get_json(), timeout=5)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"success": False, "error": "AI Service unavailable"}), 500

@app.route('/api/video_control', methods=['POST'])
def video_control():
    try:
        resp = requests.post(f"{AI_SERVICE_URL}/api/video_control", json=request.get_json(), timeout=5)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"success": False, "error": "AI Service unavailable"}), 500

@app.route('/api/available_videos')
def available_videos():
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

    try:
        resp = requests.get(f"{AI_SERVICE_URL}/api/state", timeout=2)
        ai_state = resp.json()
        cam1 = ai_state.get("cameras", {}).get("1", {})
        cam2 = ai_state.get("cameras", {}).get("2", {})
    except:
        cam1 = {}
        cam2 = {}

    return jsonify({
        "samples": samples,
        "uploads": uploads,
        "cam1": cam1,
        "cam2": cam2
    })

@app.route('/api/incidents/<int:incident_id>/status', methods=['POST'])
def update_incident_status(incident_id):
    data = request.get_json() or {}
    new_status = data.get('status', 'FALSE_ALARM')
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("UPDATE trespass_logs SET status = ? WHERE id = ?", (new_status, incident_id))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "id": incident_id, "status": new_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/internal/log_incident', methods=['POST'])
def internal_log_incident():
    data = request.get_json()
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO trespass_logs
            (timestamp, person_id, person_label, confidence, centroid_x, centroid_y, status, evidence_image, incident_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (current_time, data['person_id'], data['label'], data['confidence'], 
             data['centroid_x'], data['centroid_y'], data['status'], data.get('evidence_image'), data.get('incident_type'))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/internal/upload_evidence', methods=['POST'])
def internal_upload_evidence():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    save_path = os.path.join(config.EVIDENCE_DIR, filename)
    file.save(save_path)
    return jsonify({"success": True, "filepath": filename})

@app.route('/api/state')
def get_state():
    db_logs = []
    total_incidents = 0
    try:
        conn = sqlite3.connect(config.DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
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
            if "[CAM-02]" in label_str: cam_tag = "CAM-02"

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
        pass

    try:
        resp = requests.get(f"{AI_SERVICE_URL}/api/state", timeout=2)
        ai_state = resp.json()
        cam1_state = ai_state.get("cameras", {}).get("1", {})
        cam2_state = ai_state.get("cameras", {}).get("2", {})
        live_logs = ai_state.get("live_logs", [])
    except:
        cam1_state = {"stream_status": "OFFLINE", "fps": 0, "active_tracks": 0}
        cam2_state = {"stream_status": "OFFLINE", "fps": 0, "active_tracks": 0}
        live_logs = []

    seen_keys = set()
    combined_logs = []
    
    for item in sorted(live_logs, key=lambda x: x["time"], reverse=True):
        key = (item.get("evidence_filename") or item.get("object"), item.get("time"))
        if key not in seen_keys:
            seen_keys.add(key)
            combined_logs.append(item)

    for item in db_logs:
        key = (item.get("evidence_filename") or item.get("object"), item.get("time"))
        if key not in seen_keys:
            seen_keys.add(key)
            combined_logs.append(item)

    combined_logs = combined_logs[:25]
    active_alert = cam1_state.get("recent_alert") or cam2_state.get("recent_alert")

    return jsonify({
        "cameras": {"1": cam1_state, "2": cam2_state},
        "stream_status": cam1_state.get("stream_status", "OFFLINE"),
        "active_video": cam1_state.get("active_source"),
        "is_processing": cam1_state.get("stream_status") == "LIVE" or cam2_state.get("stream_status") == "LIVE",
        "fps": cam1_state.get("fps", 0),
        "active_tracks": cam1_state.get("active_tracks", 0) + cam2_state.get("active_tracks", 0),
        "total_incidents": total_incidents,
        "recent_alert": active_alert,
        "system_health": "ONLINE" if cam1_state.get("stream_status") != "OFFLINE" else "AI OFFLINE",
        "model": "yolo11n.pt",
        "logs": combined_logs
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 BORDER SURVEILLANCE WEB DASHBOARD LIVE: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
