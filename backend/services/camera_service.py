import os
import time
import threading
import socket
from urllib.parse import urlparse
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Ensure ffmpeg timeout options are configured on Windows
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"  # 5s timeout in microseconds

def format_video_timestamp(seconds: float) -> str:
    """Format float seconds into HH:MM:SS.ff string (e.g. 00:01:42.35)."""
    if seconds is None or seconds < 0:
        return "00:00:00.00"
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:02d}"

def get_local_lan_ips() -> List[str]:
    """Retrieve all local non-loopback IPv4 addresses assigned to laptop network interfaces."""
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    if not ips:
        # Fallback socket method
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return ips

def diagnose_network_failure(host: str, port: int, candidate_urls: List[str], err: Optional[Exception] = None) -> str:
    """
    Clearly distinguish exact failure root cause:
    - PHONE APP NOT RUNNING
    - WRONG PHONE IP
    - WRONG STREAM ENDPOINT
    - PHONE/LAPTOP NOT ACTUALLY ON SAME LAN
    - WINDOWS FIREWALL
    - ROUTER CLIENT ISOLATION
    - OPENCV HTTP STREAM COMPATIBILITY
    """
    local_ips = get_local_lan_ips()
    local_ip_str = ", ".join(local_ips) if local_ips else "unknown"

    # 1. Check for Subnet mismatch / Cellular Mobile Data IP (e.g. 192.0.0.x vs 192.168.x.x)
    if host.startswith("192.0.0.") and any(ip.startswith("192.168.") for ip in local_ips):
        return (
            f"[PHONE/LAPTOP NOT ACTUALLY ON SAME LAN] Laptop is on Wi-Fi subnet ({local_ip_str}), "
            f"but phone reported IP {host} which is a cellular Mobile Data CGNAT address. "
            f"The IP Webcam app on the phone bound to Mobile Data instead of Wi-Fi. "
            f"FIX: Turn OFF Mobile Data on your phone, keep only Wi-Fi turned ON, then restart IP Webcam. "
            f"Enter the 192.168.1.X address shown at the bottom of the IP Webcam screen."
        )

    # 2. Check if host subnet doesn't match any local interface
    host_subnet = ".".join(host.split(".")[:3]) if host.count(".") == 3 else ""
    local_subnets = [".".join(ip.split(".")[:3]) for ip in local_ips if ip.count(".") == 3]
    if host_subnet and local_subnets and host_subnet not in local_subnets:
        return (
            f"[PHONE/LAPTOP NOT ACTUALLY ON SAME LAN] Phone IP {host} is on subnet '{host_subnet}.x', "
            f"while laptop is on '{local_subnets[0]}.x' (Local IPs: {local_ip_str}). "
            f"Verify both phone and laptop are connected to the exact same Wi-Fi network and band."
        )

    # 3. Connection Refused (port closed -> host is reachable, but app is not running)
    if isinstance(err, ConnectionRefusedError) or (err and "10061" in str(err)):
        return (
            f"[PHONE APP NOT RUNNING] Phone at {host} is reachable on the network, but port {port} refused connection. "
            f"Ensure the IP Webcam app is opened on the phone and 'Start Server' at the bottom of the app is active."
        )

    # 4. Timeout error on the same subnet -> Router AP isolation or Windows Firewall
    if isinstance(err, socket.timeout) or (err and "timed out" in str(err).lower()):
        return (
            f"[ROUTER CLIENT ISOLATION or WINDOWS FIREWALL] Connection to {host}:{port} timed out. "
            f"1) Check if your Wi-Fi router has 'AP/Client Isolation' turned on (which prevents Wi-Fi devices from seeing each other). "
            f"2) Temporarily check if Windows Firewall is blocking inbound/outbound local traffic. "
            f"3) Alternatively, enable USB Tethering or phone mobile hotspot for a direct connection."
        )

    # 5. Host Unreachable (WinError 10065 or 10051)
    if err and any(code in str(err) for code in ["10065", "10051", "unreachable"]):
        return (
            f"[WRONG PHONE IP] No active device responded at IP {host} on local network ({local_ip_str}). "
            f"Check the exact IP address displayed on your phone's IP Webcam screen."
        )

    # 6. Wrong stream endpoint
    if not err and candidate_urls:
        return (
            f"[WRONG STREAM ENDPOINT] Connected to host {host}:{port}, but none of the tested stream endpoints "
            f"({', '.join(candidate_urls)}) returned a valid MJPEG video stream."
        )

    return f"NETWORK ERROR: {str(err) if err else 'Could not connect to camera stream'}"

class CameraState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAM_ERROR = "stream_error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

class CameraSource:
    """
    Robust Camera Stream Abstraction with State Machine & Diagnostics:
    States: DISCONNECTED, CONNECTING, CONNECTED, STREAM_ERROR, RECONNECTING, STOPPED.
    
    Supports:
    1. Phone IP Webcam (HTTP/MJPEG/RTSP) with smart endpoint auto-discovery (/video, /videofeed, /mjpeg)
    2. Local MP4 video file with frame indices and video_timestamp calculation
    3. Local webcam (0, 1, etc.)
    """
    def __init__(self, source: str = "video.mp4", loop_video: bool = True):
        self.raw_source = str(source).strip()
        self.loop_video = loop_video
        self.cap: Optional[cv2.VideoCapture] = None
        
        # State machine
        self.state = CameraState.DISCONNECTED
        self.error_message: Optional[str] = None
        
        # Metadata
        self.source_type = "video_file"
        self.fps = 30.0
        self.frame_count = 0
        self.frames_received = 0
        self.frame_index = 0
        self.width = 0
        self.height = 0
        self.last_frame_time: Optional[float] = None
        self.last_reconnect_attempt = 0.0
        self.reconnect_interval = 3.0  # seconds
        
        # Thread-safe frame buffer for live preview/MJPEG streaming
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id: int = 0
        self.latest_video_ts: Optional[str] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False
        
        # Camera Location Metadata & Priority State
        # Priority: 1. gps > 2. registered > 3. ip > unknown
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.location_source: str = "unknown"
        self.location_accuracy: Optional[float] = None
        self.location_timestamp: Optional[str] = None
        self._init_registered_location()

        self.candidates: List[str] = []
        self._classify_and_prepare_candidates()

    def _init_registered_location(self):
        """Load registered camera coordinates from environment if configured."""
        reg_lat = os.getenv("REGISTERED_LATITUDE", "").strip()
        reg_lon = os.getenv("REGISTERED_LONGITUDE", "").strip()
        if reg_lat and reg_lon:
            try:
                self.latitude = float(reg_lat)
                self.longitude = float(reg_lon)
                self.location_source = "registered"
                self.location_accuracy = 50.0  # nominal fixed CCTV precision
                self.location_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(f"[LOCATION] Registered coordinates loaded: {self.latitude}, {self.longitude} (source=registered)")
            except ValueError:
                pass

    def update_location(self, lat: float, lon: float, source: str = "gps", accuracy: Optional[float] = None) -> Dict[str, Any]:
        """
        Update camera location enforcing strict source priority:
        1. GPS (phone device GPS with user consent) - highest priority, overrides all
        2. Registered (admin configured fixed coordinates) - overrides IP/unknown
        3. IP Geolocation - fallback only; never overrides GPS or registered
        """
        src = str(source).lower().strip()
        if src not in ["gps", "registered", "ip"]:
            src = "unknown"

        # Priority rules
        if self.location_source == "gps" and src != "gps":
            print(f"[LOCATION] Rejected update from '{src}': active GPS has higher priority")
            return self.get_location()

        if self.location_source == "registered" and src == "ip":
            print(f"[LOCATION] Rejected IP fallback update: registered location has higher priority")
            return self.get_location()

        self.latitude = float(lat)
        self.longitude = float(lon)
        self.location_source = src
        self.location_accuracy = float(accuracy) if accuracy is not None else None
        self.location_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        print(f"[LOCATION] Camera location updated: {self.latitude:.5f}, {self.longitude:.5f} (source={self.location_source}, accuracy={self.location_accuracy}m)")
        return self.get_location()

    def get_location(self) -> Dict[str, Any]:
        """Return current camera location metadata."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "location_accuracy": self.location_accuracy,
            "timestamp": self.location_timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

    def _classify_and_prepare_candidates(self):
        """Classify source and compute endpoint candidates if base URL is provided."""
        s = self.raw_source
        if s.isdigit():
            self.source = int(s)
            self.source_type = "webcam"
            self.is_video_file = False
            self.candidates = [str(s)]
        elif s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://"):
            self.source_type = "ip_camera"
            self.is_video_file = False
            
            parsed = urlparse(s)
            host = parsed.hostname or ""
            port = parsed.port or (443 if parsed.scheme == "https" else (554 if parsed.scheme == "rtsp" else 80))
            path = parsed.path.strip()

            # If user entered only the base IP Webcam URL (path is empty or "/"),
            # automatically try the standard IP Webcam stream endpoints in priority order:
            if not path or path == "/":
                base = f"{parsed.scheme}://{host}:{port}"
                self.candidates = [
                    f"{base}/video",       # Standard Android IP Webcam MJPEG endpoint
                    f"{base}/videofeed",   # DroidCam / alternative MJPEG endpoint
                    f"{base}/mjpeg",       # Generic MJPEG endpoint
                    f"{base}/shot.jpg",    # Single JPEG snapshot stream
                    base                   # Direct base URL fallback
                ]
                self.source = self.candidates[0]
            else:
                # User provided an explicit endpoint (e.g. /video, /videofeed, /live.mjpg) -> DO NOT guess
                self.candidates = [s]
                self.source = s
        else:
            # File path: prevent path traversal
            clean_name = os.path.basename(s)
            normalized = os.path.normpath(s)
            if ".." in normalized or normalized.startswith("/") or normalized.startswith("\\"):
                normalized = clean_name
            self.source = normalized
            self.source_type = "video_file"
            self.is_video_file = True
            self.candidates = [normalized]

    def _test_and_open_ip_camera(self) -> bool:
        """
        Sequentially test network reachability, endpoints, OpenCV VideoCapture,
        and frame decoding with full terminal diagnostics.
        """
        parsed = urlparse(self.raw_source)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        print(f"[PHONE CAM] URL: {self.raw_source}")
        print(f"[PHONE CAM] Testing network connection...")

        # Step 1: TCP Socket Reachability Test
        try:
            with socket.create_connection((host, port), timeout=1.5):
                pass
            print(f"[PHONE CAM] Endpoint reachable: YES")
        except Exception as sock_err:
            print(f"[PHONE CAM] Endpoint reachable: NO")
            print(f"[PHONE CAM] OpenCV opened: NO")
            print(f"[PHONE CAM] Frame received: NO")
            diag = diagnose_network_failure(host, port, self.candidates, sock_err)
            print(f"[PHONE CAM] ERROR: {diag}")
            print(f"[CAMERA] Error: {diag}")
            self.state = CameraState.DISCONNECTED
            self.error_message = diag
            return False

        # Step 2: Test Candidates for working video stream
        print(f"[PHONE CAM] Probing stream endpoints: {self.candidates}")
        working_candidate = None
        working_cap = None
        working_frame = None

        for candidate in self.candidates:
            try:
                cap = cv2.VideoCapture(candidate)
                if not cap or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue

                # Buffer size 1 to avoid lag / internal frame backlog
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Test reading actual frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    working_candidate = candidate
                    working_cap = cap
                    working_frame = frame
                    print(f"[PHONE CAM] Found working endpoint: {candidate}")
                    break
                else:
                    cap.release()
            except Exception:
                pass

        if working_cap is None or working_frame is None:
            print(f"[PHONE CAM] OpenCV opened: NO")
            print(f"[PHONE CAM] Frame received: NO")
            diag = diagnose_network_failure(host, port, self.candidates, None)
            print(f"[PHONE CAM] ERROR: {diag}")
            print(f"[CAMERA] Error: {diag}")
            self.state = CameraState.DISCONNECTED
            self.error_message = diag
            return False

        # Success! Connected and read valid frame
        self.cap = working_cap
        self.source = working_candidate
        self.state = CameraState.CONNECTED
        self.error_message = None

        self.height, self.width = working_frame.shape[:2]
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_val if (fps_val and fps_val > 0) else 25.0
        self.frames_received = 1
        self.last_frame_time = time.time()

        with self.frame_lock:
            self.latest_frame = working_frame.copy()
            self.latest_frame_id += 1
            self.latest_annotated_frame = working_frame.copy()

        print(f"[PHONE CAM] OpenCV opened: YES")
        print(f"[PHONE CAM] Frame received: YES")
        print(f"[PHONE CAM] Resolution: {self.width}x{self.height}")
        print(f"[PHONE CAM] FPS: {int(round(self.fps))}")
        print(f"[CAMERA] Connected: Resolution {self.width}x{self.height} @ {self.fps:.1f} FPS (Source: {working_candidate})")
        return True

    def open(self) -> bool:
        """
        Open the video capture device or stream.
        Starts the independent background capture thread once connection is verified.
        """
        self._stop_capture_thread()
        self.state = CameraState.CONNECTING
        self.error_message = None
        
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        print(f"[CAMERA] Connecting to {self.source_type}: '{self.source}'...")

        if self.source_type == "ip_camera":
            connected = self._test_and_open_ip_camera()
        else:
            # Fast existence check for local video files
            if self.source_type == "video_file" and not os.path.exists(str(self.source)):
                self.state = CameraState.DISCONNECTED
                self.error_message = f"Video file '{self.source}' not found."
                print(f"[CAMERA] Error: {self.error_message}")
                return False

            try:
                if self.source_type == "webcam":
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.source)
            except Exception as e:
                self.state = CameraState.DISCONNECTED
                self.error_message = f"Failed to initialize VideoCapture: {str(e)}"
                print(f"[CAMERA] Error: {self.error_message}")
                return False

            if not self.cap or not self.cap.isOpened():
                self.state = CameraState.DISCONNECTED
                self.error_message = f"Cannot open video source '{self.source}'"
                print(f"[CAMERA] Error: {self.error_message}")
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # REAL TEST: Attempt to actually read a frame before claiming CONNECTED
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                self.state = CameraState.DISCONNECTED
                self.error_message = f"Opened source '{self.source}' but failed to read initial frame."
                print(f"[CAMERA] Error: {self.error_message}")
                self.cap.release()
                self.cap = None
                return False

            # Success: valid frame received!
            self.state = CameraState.CONNECTED
            self.error_message = None
            
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps_val if (fps_val and fps_val > 0) else 30.0
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_video_file else 0
            self.height, self.width = test_frame.shape[:2]
            self.frames_received = 1
            self.last_frame_time = time.time()
            
            with self.frame_lock:
                self.latest_frame = test_frame.copy()
                self.latest_frame_id += 1
                self.latest_annotated_frame = test_frame.copy()

            print(f"[CAMERA] Frame received")
            print(f"[CAMERA] Connected: Resolution {self.width}x{self.height} @ {self.fps:.1f} FPS (Source: {self.source})")
            connected = True

        if connected:
            self._start_capture_thread()
        return connected

    def _start_capture_thread(self):
        """Start background camera reader thread."""
        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self._capture_thread.start()
        print("[CAMERA] Dedicated background capture thread started.")

    def _stop_capture_thread(self):
        """Stop background camera reader thread."""
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self._capture_thread = None

    def _capture_loop(self):
        """
        Continuous, non-blocking camera frame capture loop.
        Drains frames at full camera frame-rate into a single thread-safe latest_frame buffer.
        Never queues frames; always overwrites with the newest frame to prevent latency drift.
        """
        while self._capture_running:
            if self.state != CameraState.CONNECTED or self.cap is None:
                # Auto-reconnection logic
                now = time.time()
                if (now - self.last_reconnect_attempt) > self.reconnect_interval:
                    self.last_reconnect_attempt = now
                    self.state = CameraState.RECONNECTING
                    print(f"[CAMERA] Reconnecting to '{self.source}'...")
                    if self.source_type == "ip_camera":
                        ok = self._test_and_open_ip_camera()
                    else:
                        ok = self.open()
                    if ok:
                        print(f"[CAMERA] Reconnection successful!")
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(0.2)
                continue

            loop_start = time.time()
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_video_file and self.loop_video and self.frame_count > 0:
                    # Video file reached end -> rewind to frame 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_index = 0
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self.state = CameraState.DISCONNECTED
                        self.error_message = "Video file playback ended."
                        time.sleep(0.5)
                        continue
                else:
                    # Live stream disconnected or frame read failure
                    self.state = CameraState.STREAM_ERROR
                    self.error_message = "Camera stream frame read failed. Auto-reconnecting..."
                    print(f"[CAMERA] Read failure. State set to STREAM_ERROR.")
                    time.sleep(0.5)
                    continue

            self.frame_index += 1
            self.frames_received += 1
            self.last_frame_time = time.time()

            # Calculate exact video timestamp from frame index / FPS for video files
            if self.is_video_file and self.fps > 0:
                vid_secs = self.frame_index / float(self.fps)
                video_ts = format_video_timestamp(vid_secs)
            else:
                video_ts = None  # Live streams use wall-clock ISO timestamps

            # Overwrite single latest frame buffer under lock
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.latest_video_ts = video_ts

            # Frame pacing:
            if self.is_video_file and self.fps > 0:
                frame_interval = 1.0 / self.fps
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)
            else:
                # Live stream: cap.read() blocks for next camera frame; tiny yield to prevent thread lock
                time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int, Optional[str]]:
        """
        Thread-safe read of the latest captured frame.
        Maintains backwards compatibility for any synchronous callers.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return False, None, self.frame_index, None
            return True, self.latest_frame.copy(), self.frame_index, self.latest_video_ts

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
        """
        Retrieve the latest frame, frame ID, and video timestamp safely without blocking.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None, 0, None
            return self.latest_frame.copy(), self.latest_frame_id, self.latest_video_ts

    def get_latest_frame_with_id(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Fast access to the latest frame and ID for continuous streaming.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_frame_id

    def set_annotated_frame(self, frame: np.ndarray):
        """Set latest annotated frame for live streaming with HUD overlays."""
        with self.frame_lock:
            self.latest_annotated_frame = frame.copy()

    def get_latest_jpeg(self, use_annotated: bool = True) -> Optional[bytes]:
        """Encode latest frame as JPEG bytes for HTTP preview."""
        with self.frame_lock:
            target = self.latest_annotated_frame if (use_annotated and self.latest_annotated_frame is not None) else self.latest_frame
            if target is None:
                return None
            ret, buffer = cv2.imencode('.jpg', target, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buffer.tobytes() if ret else None

    def release(self):
        """Release capture device completely."""
        self._stop_capture_thread()
        self.state = CameraState.STOPPED
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[CAMERA] Stream released.")

    def get_status(self) -> Dict[str, Any]:
        """
        Return rich real-time camera status matching specification:
        {
          "status": "CONNECTED",
          "url": "http://192.0.0.2:8080/video",
          "resolution": "1280x720",
          "fps": 24,
          "frames_received": 1234,
          "error": null
        }
        """
        last_time_str = None
        if self.last_frame_time:
            last_time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_frame_time))

        res_str = f"{self.width}x{self.height}" if (self.width > 0 and self.height > 0) else "0x0"
        status_str = "CONNECTED" if self.state == CameraState.CONNECTED else (
            "CONNECTING" if self.state in [CameraState.CONNECTING, CameraState.RECONNECTING] else "DISCONNECTED"
        )

        return {
            "status": status_str,
            "url": str(self.source),
            "resolution": res_str,
            "fps": int(round(self.fps)) if self.fps else 0,
            "frames_received": self.frames_received,
            "error": self.error_message,
            # Backwards-compatibility metadata for UI/WebSocket
            "camera_id": "PHONE_CAM_01",
            "source": self.raw_source,
            "source_type": self.source_type,
            "current_frame": self.frame_index,
            "last_frame_time": last_time_str,
            "normalized_source": str(self.source),
            # Camera Location Metadata
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "location_accuracy": self.location_accuracy,
            "location_label": (
                f"GPS (±{int(self.location_accuracy or 10)}m)" if self.location_source == "gps" else (
                    "Registered Coordinates" if self.location_source == "registered" else (
                        "Approximate location (Source: IP geolocation)" if self.location_source == "ip" else (
                            "Coordinates: null (Single Sensor)"
                        )
                    )
                )
            )
        }
