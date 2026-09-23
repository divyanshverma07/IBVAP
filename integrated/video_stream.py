import cv2

class VideoStream:
    def __init__(self, source, camera_id="CAM-MAIN", max_dim=960):
        self.raw_source = source
        self.camera_id = camera_id
        self.max_dim = max_dim
        self.cap = None
        self.fps = 30
        self.width = 960
        self.height = 540
        self.skip_stride = 1
        self.is_live = False

        # Normalize source (int 0 or string "0" / "live" -> webcam 0)
        if isinstance(source, str) and (source.strip().lower() == "live" or source.strip().isdigit()):
            self.source = int(source) if source.strip().isdigit() else 0
            self.is_live = True
        elif isinstance(source, int):
            self.source = source
            self.is_live = True
        else:
            self.source = source
            self.is_live = False

    def connect(self):
        if self.is_live:
            # On Windows, try cv2.CAP_DSHOW for rapid webcam startup
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
            # If physical webcam is not detected on device, use sample video as simulated live stream
            if not self.cap or not self.cap.isOpened():
                import os
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                sim_clip = os.path.join(base_dir, "test6.mp4")
                if os.path.exists(sim_clip):
                    print(f"[ONLINE] {self.camera_id} - Physical camera device {self.source} not found. Running simulated live camera.")
                    self.cap = cv2.VideoCapture(sim_clip)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] {self.camera_id} connection failed to {self.raw_source}")
            return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps > 120:
            self.fps = 30
            
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w <= 0: orig_w = 640
        if orig_h <= 0: orig_h = 480
        
        # If source is 50-60 FPS, skip every 2nd frame so it plays in real-time
        if self.fps >= 50 and not self.is_live:
            self.skip_stride = 2
        else:
            self.skip_stride = 1
            
        # Scale to max_dim for real-time multi-camera CPU performance
        max_orig = max(orig_w, orig_h)
        if max_orig > self.max_dim:
            scale = self.max_dim / max_orig
            self.width = int(orig_w * scale)
            self.height = int(orig_h * scale)
        else:
            self.width = orig_w
            self.height = orig_h
        
        src_label = f"LiveWebcam({self.source})" if self.is_live else str(self.raw_source)
        print(f"[ONLINE] {self.camera_id} - Source: {src_label} [{orig_w}x{orig_h}@{self.fps:.0f}FPS] -> Stream: {self.width}x{self.height} (skip={self.skip_stride})")
        return True

    def read_frame(self):
        if self.cap is None:
            return False, None
            
        # Fast grab skip for high FPS videos to prevent slow-motion playback
        if self.skip_stride > 1 and not self.is_live:
            for _ in range(self.skip_stride - 1):
                self.cap.grab()
                
        ret, frame = self.cap.read()
        if ret and frame is not None and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print(f"[OFFLINE] {self.camera_id}")

