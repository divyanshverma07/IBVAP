import os
import re
import sys
import time
import subprocess
import threading

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
sys.path.insert(0, cur_dir)

cloudflared_path = os.path.abspath(os.path.join(cur_dir, "..", "cloudflared.exe"))

def start_tunnel():
    if not os.path.exists(cloudflared_path):
        print(f"[ERROR] cloudflared.exe not found at {cloudflared_path}")
        return

    process = subprocess.Popen(
        [cloudflared_path, "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url_found = False
    for line in iter(process.stdout.readline, ''):
        match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
        if match and not url_found:
            url = match.group(1)
            print("\n" + "="*70)
            print("🚀 YOUR LIVE PROJECT IS ONLINE! CLICK OR COPY THIS LINK:")
            print(f"👉  {url}  👈")
            print("="*70 + "\n", flush=True)
            url_found = True

t_tunnel = threading.Thread(target=start_tunnel, daemon=True)
t_tunnel.start()

import app
os.makedirs(app.config.EVIDENCE_DIR, exist_ok=True)
t_ai = threading.Thread(target=app.processing_thread, daemon=True)
t_ai.start()

print("🚀 Starting Flask web dashboard on http://127.0.0.1:5000 ...", flush=True)
app.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
