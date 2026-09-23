import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import threading
from pyngrok import ngrok

NGROK_TOKEN = "3ItsxQwGLVlqVMC58p7M4fxSyNi_6cFpF3qymbVZQZBYwC6CH"

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
sys.path.insert(0, cur_dir)

try:
    print("[*] Connecting ngrok tunnel to port 5000...")
    ngrok.set_auth_token(NGROK_TOKEN)
    tunnel = ngrok.connect(5000)
    print("\n" + "="*65)
    print(f"🚀 IBVAP IS LIVE ON NGROK: {tunnel.public_url}")
    print("="*65 + "\n")
except Exception as e:
    print(f"\n[!] Failed to start ngrok: {e}")
    print("If Windows Defender blocked ngrok.exe:")
    print("  1. Open Windows Security -> Virus & threat protection -> Protection history")
    print("  2. Find ngrok.exe and click Actions -> 'Allow on device'")
    sys.exit(1)

import app
os.makedirs(app.config.EVIDENCE_DIR, exist_ok=True)
t = threading.Thread(target=app.processing_thread, daemon=True)
t.start()

print("🚀 Starting Flask web dashboard on http://127.0.0.1:5000")
app.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
