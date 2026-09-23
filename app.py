import sys
import os
import threading

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the integrated folder to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integrated'))

# Import the Flask app and the background thread from integrated/app.py
from integrated.app import app, processing_thread
import integrated.config as config

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "integrated", "uploads"), exist_ok=True)
    
    # Start the background video AI thread
    t = threading.Thread(target=processing_thread, daemon=True)
    t.start()
    
    # Hugging Face Spaces strictly requires apps to run on port 7860
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 STARTING FLASK ON HUGGING FACE (PORT {port}) 🚀\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
