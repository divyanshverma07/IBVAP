import os
import sys
import subprocess
import threading
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
sys.path.insert(0, cur_dir)

def start_tunnel():
    time.sleep(2)
    print("\n" + "="*65)
    print("🌍 STARTING PUBLIC TUNNEL VIA WINDOWS OPENSSH...")
    print("="*65)
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-R", "80:localhost:5000", "nokey@localhost.run"]
    subprocess.run(cmd)

t_tunnel = threading.Thread(target=start_tunnel, daemon=True)
t_tunnel.start()

import app
os.makedirs(app.config.EVIDENCE_DIR, exist_ok=True)
t_ai = threading.Thread(target=app.processing_thread, daemon=True)
t_ai.start()

print("🚀 Starting Flask web dashboard on http://127.0.0.1:5000")
app.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
