import os
import time
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from backend.inference.detector import ANPRDetector
from backend.inference.tracker import PlateTracker
from backend.inference.ocr import PlateOCR, normalize_plate_text, calculate_plate_similarity
from backend.services.camera_service import CameraSource, CameraState
from backend.api.websocket import ws_manager
from backend.models.schemas import DetectionEvent, TargetDetectionEvent, PipelineStateEvent, SearchResponse
from backend.db.database import insert_detection

class DetectionService:
    """
    Main ANPR inference pipeline coordinator with Target Plate Matching.
    Flow:
      Camera frame -> YOLO best.pt (license-plate detector) -> bbox -> crop
                   -> PlateTracker (track_id) -> PlateOCR -> Normalize text
                   -> Fuzzy similarity comparison with active target plate
                   -> If MATCH: save actual plate crop & annotated frame evidence
                                -> broadcast target_detection event
                   -> If NO MATCH: ignore as target result
    """
    def __init__(
        self,
        weights_path: str = "best.pt",
        camera_source: str = "video.mp4",
        camera_id: str = "PHONE_CAM_01",
        cooldown_seconds: float = 2.0,
        similarity_threshold: float = 0.85,
        debug_detections: bool = False
    ):
        self.camera_id = camera_id
        self.camera_name = "Traffic Sensor Node"
        self.cooldown_seconds = cooldown_seconds
        self.similarity_threshold = similarity_threshold
        self.debug_detections = debug_detections

        # Output directories for genuine model-generated evidence
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.plates_dir = os.path.join(base_dir, "detections", "plates")
        self.frames_dir = os.path.join(base_dir, "detections", "frames")
        os.makedirs(self.plates_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        # Initialize detector, tracker, OCR
        self.detector = ANPRDetector(weights_path=weights_path, conf_threshold=0.25)
        self.tracker = PlateTracker(iou_threshold=0.35)
        self.ocr = PlateOCR(use_gpu=True, min_confidence=0.30)

        # Camera source with state machine
        self.camera_source_str = camera_source
        self.camera = CameraSource(source=camera_source, loop_video=True)

        # Pipeline state and search session
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.detection_counter = 0

        # Duplicate control: {normalized_plate: last_broadcast_timestamp}
        self.plate_cooldowns: Dict[str, float] = {}

        # Search session tracking: {session_id: SearchResponse}
        self.active_search: Optional[SearchResponse] = None
        self.target_detections: List[Dict[str, Any]] = []
        self.last_target_match: Optional[Dict[str, Any]] = None

        # Chronological in-memory storage: {normalized_plate: [DetectionEvent]}
        self.vehicle_records: Dict[str, List[DetectionEvent]] = {}
        self.all_detections: List[DetectionEvent] = []
        self._last_broadcast_cam_state: Optional[str] = None

        # Thread-safe detection overlay buffer
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_detections_time: float = 0.0
        self.detections_lock = threading.Lock()

    def get_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Lightweight overlay of active detection bounding boxes onto a frame.
        Runs in microseconds/milliseconds without blocking streaming.
        """
        with self.detections_lock:
            dets = self.latest_detections
            det_time = self.latest_detections_time

        if dets and (time.time() - det_time) < 0.8:
            try:
                return self.detector.draw_detections(frame.copy(), dets)
            except Exception:
                return frame
        return frame

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.event_loop = loop

    def set_search_target(self, target_plate: str, state: str = "Delhi") -> SearchResponse:
        """Create or update active search session for a target plate."""
        normalized = normalize_plate_text(target_plate)
        session_id = f"search_{int(time.time())}_{normalized}"
        session = SearchResponse(
            session_id=session_id,
            target_plate=normalized,
            normalized_plate=normalized,
            state=state,
            status="searching"
        )
        self.active_search = session
        self.target_detections = []
        self.last_target_match = None
        print(f"[SEARCH] Active search target set: '{normalized}' ({state}) (threshold={self.similarity_threshold})")
        
        # Broadcast search start event
        self._dispatch_async(ws_manager.broadcast({
            "type": "search_session_started",
            "session_id": session_id,
            "target_plate": target_plate,
            "normalized_plate": normalized,
            "state": state
        }))
        return session

    def get_target_status(self) -> Dict[str, Any]:
        """Return current target search metadata, status, and matches."""
        if not self.active_search:
            return {
                "active": False,
                "status": "NOT_SET",
                "target_plate": None,
                "state": None,
                "detections": [],
                "last_match": None
            }
        return {
            "active": True,
            "status": self.active_search.status.upper(),
            "target_plate": self.active_search.target_plate,
            "normalized_plate": self.active_search.normalized_plate,
            "state": self.active_search.state,
            "target_state": self.active_search.state,
            "total_matches": len(self.target_detections),
            "detections": self.target_detections,
            "last_match": self.last_target_match
        }

    def update_camera_source(self, new_source: str, new_camera_id: str = None) -> Dict[str, Any]:
        """Dynamically switch camera source (e.g. to phone IP stream or MP4)."""
        print(f"\n[CAMERA] Updating camera source to: '{new_source}'")
        self.camera_source_str = new_source
        if new_camera_id:
            self.camera_id = new_camera_id

        # Re-initialize camera
        self.camera.release()
        self.camera = CameraSource(source=new_source, loop_video=True)
        connected = self.camera.open()

        self._broadcast_camera_status()
        return self.camera.get_status()

    def reconnect_camera(self) -> Dict[str, Any]:
        """Force a manual reconnection attempt."""
        print(f"[CAMERA] Manual reconnect triggered for '{self.camera.source}'...")
        self.camera.open()
        self._broadcast_camera_status()
        return self.camera.get_status()

    def update_camera_location(self, lat: float, lon: float, source: str = "gps", accuracy: Optional[float] = None) -> Dict[str, Any]:
        """Update active camera location with priority handling and broadcast status."""
        if not self.camera:
            return {"status": "error", "message": "Camera not initialized"}
        loc = self.camera.update_location(lat, lon, source, accuracy)
        self._broadcast_camera_status()
        return {
            "camera_id": self.camera_id,
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "location_source": loc["location_source"],
            "location_accuracy": loc["location_accuracy"],
            "timestamp": loc["timestamp"]
        }

    def start(self):
        """Start the background inference processing thread."""
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()
        print("[AI ENGINE] Background inference thread started.")

    def stop(self):
        """Stop inference processing."""
        self.is_running = False
        if self.camera:
            self.camera.release()
        print("[AI ENGINE] Inference thread stopped.")

    def _dispatch_async(self, coro):
        """Thread-safe dispatch to FastAPI's asyncio event loop."""
        if self.event_loop and self.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.event_loop)

    def _broadcast_camera_status(self):
        """Broadcast current camera status over WebSocket."""
        status = self.camera.get_status()
        self._dispatch_async(ws_manager.broadcast({
            "type": "camera_status",
            **status
        }))

    def _broadcast_state(self, state: str, message: str, details: Optional[dict] = None):
        """Helper to broadcast pipeline state changes."""
        evt = PipelineStateEvent(state=state, message=message, details=details)
        self._dispatch_async(ws_manager.broadcast(evt.model_dump()))

    def _inference_loop(self):
        """
        Independent AI inference worker:
        - Decoupled from camera capture thread.
        - Fetches single newest frame from CameraSource.
        - Discards old/intermediate frames (zero latency drift).
        - Executes YOLO detection and OCR at controlled inference rate (~8-12 FPS).
        - Updates detection boxes & target matching without delaying camera stream.
        """
        self._broadcast_camera_status()
        self.camera.open()
        self._broadcast_camera_status()

        target_interval = 0.08  # Max ~12 FPS for YOLO inference
        last_inference_time = 0.0
        last_processed_frame_id = -1

        while self.is_running:
            # Broadcast state change if camera connection status changed
            if self.camera.state != self._last_broadcast_cam_state:
                self._last_broadcast_cam_state = self.camera.state
                self._broadcast_camera_status()

            # Pacing: Avoid running faster than target inference rate
            now = time.time()
            elapsed_since_last = now - last_inference_time
            if elapsed_since_last < target_interval:
                time.sleep(min(0.015, target_interval - elapsed_since_last))
                continue

            if self.camera.state != CameraState.CONNECTED:
                time.sleep(0.2)
                continue

            # Grab newest frame from thread-safe camera buffer
            frame, frame_id, video_ts = self.camera.get_latest_frame()
            if frame is None or frame_id == last_processed_frame_id:
                time.sleep(0.005)
                continue

            last_processed_frame_id = frame_id
            last_inference_time = time.time()
            frame_num = frame_id

            # Run YOLO plate detector
            detections = self.detector.detect(frame)

            if detections:
                # Update plate tracks
                tracked_detections = self.tracker.update(detections)
                with self.detections_lock:
                    self.latest_detections = tracked_detections
                    self.latest_detections_time = time.time()

                for det in tracked_detections:
                    x1, y1, x2, y2 = det["bbox"]
                    plate_conf = det["confidence"]
                    track_id = det.get("track_id")

                    if self.debug_detections:
                        print(f"[YOLO] Plate detected confidence={plate_conf:.2f} bbox=[{x1},{y1},{x2},{y2}] (track_id={track_id})")

                    # Crop plate
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop is None or plate_crop.size == 0:
                        continue

                    # Run EasyOCR
                    if self.debug_detections:
                        print("[OCR] Processing plate crop...")
                    plate_text, ocr_conf = self.ocr.read_plate(plate_crop)
                    det["plate_text"] = plate_text
                    det["ocr_confidence"] = ocr_conf

                    if self.debug_detections:
                        if plate_text:
                            print(f"[OCR] Result: '{plate_text}' confidence={ocr_conf:.2f}")
                        else:
                            print(f"[OCR] Result: uncertain (confidence={ocr_conf}) — zero hallucination")

                    # Location metadata from camera
                    cam_lat = self.camera.latitude if self.camera else None
                    cam_lon = self.camera.longitude if self.camera else None
                    cam_loc_src = self.camera.location_source if self.camera else "unknown"
                    cam_loc_acc = self.camera.location_accuracy if self.camera else None

                    # OCR & Target Matching
                    if self.active_search:
                        if plate_text:
                            match_score = calculate_plate_similarity(self.active_search.target_plate, plate_text)
                            is_match = (match_score >= self.similarity_threshold)
                            if self.debug_detections:
                                print(f"[TARGET CHECK] OCR='{plate_text}' Target='{self.active_search.target_plate}' Similarity={match_score:.2f} (Threshold={self.similarity_threshold}) -> Match={is_match}")
                        else:
                            match_score = 0.0
                            is_match = False
                            if self.debug_detections:
                                print(f"[TARGET CHECK] OCR uncertain (low confidence or unreadable) — marked OCR_UNCERTAIN (no match)")

                        # 2. ONLY FLAG THE TARGET: Do NOT show non-matching plates as target results
                        if not is_match:
                            continue

                        # Target match confirmed!
                        print(f"🔥 [TARGET MATCH] Target plate confirmed! Plate: '{plate_text}' ~ Target: '{self.active_search.target_plate}' (Score={match_score:.2f})")

                        # Duplicate control for target alerts
                        cooldown_key = f"target_{normalize_plate_text(plate_text)}"
                        now_ts = time.time()
                        last_seen = self.plate_cooldowns.get(cooldown_key, 0.0)
                        if (now_ts - last_seen) < self.cooldown_seconds:
                            continue

                        self.plate_cooldowns[cooldown_key] = now_ts
                        self.detection_counter += 1
                        det_id = f"det_{self.detection_counter:06d}"

                        # 4. REAL EVIDENCE: Save actual plate crop and actual annotated camera frame
                        plate_filename = f"{det_id}.jpg"
                        frame_filename = f"{det_id}.jpg"
                        plate_disk_path = os.path.join(self.plates_dir, plate_filename)
                        frame_disk_path = os.path.join(self.frames_dir, frame_filename)

                        # Save genuine plate crop
                        cv2.imwrite(plate_disk_path, plate_crop)
                        print(f"[SAVED EVIDENCE] detections/plates/{plate_filename}")

                        # Save actual annotated frame with YOLO bounding box
                        frame_with_det = self.detector.draw_detections(frame, [det])
                        cv2.imwrite(frame_disk_path, frame_with_det)
                        print(f"[SAVED EVIDENCE] detections/frames/{frame_filename}")

                        plate_url = f"/detections/plates/{plate_filename}"
                        frame_url = f"/detections/frames/{frame_filename}"
                        now_iso = datetime.now().isoformat()

                        # Construct target detection payload with camera location
                        target_event = {
                            "type": "target_detection",
                            "detection_id": det_id,
                            "plate_text": plate_text,
                            "target_plate": self.active_search.target_plate,
                            "match_score": match_score,
                            "plate_confidence": round(float(plate_conf), 4),
                            "ocr_confidence": round(float(ocr_conf), 4),
                            "plate_image_url": plate_url,
                            "frame_image_url": frame_url,
                            "camera_id": self.camera_id,
                            "timestamp": now_iso,
                            "is_target_match": True,
                            "latitude": cam_lat,
                            "longitude": cam_lon,
                            "location_source": cam_loc_src,
                            "location_accuracy": cam_loc_acc,
                            "track_id": track_id,
                            "video_timestamp": video_ts
                        }

                        self.target_detections.append(target_event)
                        self.last_target_match = target_event
                        self.active_search.status = "found"

                        det_event_obj = DetectionEvent(
                            type="detection",
                            detection_id=det_id,
                            track_id=track_id,
                            plate_track_id=track_id,
                            plate_text=plate_text,
                            target_plate=self.active_search.target_plate,
                            match_score=match_score,
                            plate_confidence=round(float(plate_conf), 4),
                            ocr_confidence=round(float(ocr_conf), 4),
                            frame_number=frame_num,
                            video_timestamp=video_ts,
                            timestamp=now_iso,
                            camera_id=self.camera_id,
                            latitude=cam_lat,
                            longitude=cam_lon,
                            location_source=cam_loc_src,
                            location_accuracy=cam_loc_acc,
                            plate_image_url=plate_url,
                            frame_image_url=frame_url,
                            is_target_match=True
                        )
                        self.all_detections.append(det_event_obj)
                        norm_text = normalize_plate_text(plate_text)
                        self.vehicle_records.setdefault(norm_text, []).append(det_event_obj)

                        # Persist Target Detection to SQLite Database
                        try:
                            insert_detection({
                                "plate_text": norm_text,
                                "target_plate": self.active_search.target_plate,
                                "match_score": match_score,
                                "plate_confidence": plate_conf,
                                "ocr_confidence": ocr_conf,
                                "camera_id": self.camera_id,
                                "timestamp": now_iso,
                                "video_timestamp": video_ts,
                                "latitude": cam_lat,
                                "longitude": cam_lon,
                                "location_source": cam_loc_src,
                                "location_accuracy": cam_loc_acc,
                                "plate_image_path": plate_url,
                                "frame_image_path": frame_url
                            })
                            print(f"[DB] Persisted target detection for '{norm_text}' into SQLite")
                        except Exception as db_err:
                            print(f"[DB] Warning: Failed to persist detection to SQLite: {db_err}")

                        # Broadcast target detection to connected frontend
                        self._dispatch_async(ws_manager.broadcast(target_event))
                        self._broadcast_state(
                            "plate_match",
                            f"TARGET VEHICLE MATCH! Plate '{plate_text}' identified on camera {self.camera_id}",
                            details=target_event
                        )
                        print(f"[WS] Broadcasted target_detection: {det_id} ({plate_text})")

                    else:
                        # No active target search set
                        if not self.debug_detections:
                            continue

                        # Optional monitoring when debug_detections is enabled
                        cooldown_key = plate_text if plate_text else f"track_{track_id}"
                        now_ts = time.time()
                        if (now_ts - self.plate_cooldowns.get(cooldown_key, 0.0)) < self.cooldown_seconds:
                            continue
                        self.plate_cooldowns[cooldown_key] = now_ts

                        self.detection_counter += 1
                        det_id = f"det_{self.detection_counter:06d}"
                        plate_filename = f"{det_id}.jpg"
                        frame_filename = f"{det_id}.jpg"
                        cv2.imwrite(os.path.join(self.plates_dir, plate_filename), plate_crop)
                        frame_with_det = self.detector.draw_detections(frame, [det])
                        cv2.imwrite(os.path.join(self.frames_dir, frame_filename), frame_with_det)

                        event = DetectionEvent(
                            type="detection",
                            detection_id=det_id,
                            track_id=track_id,
                            plate_track_id=track_id,
                            plate_text=plate_text,
                            plate_confidence=round(float(plate_conf), 4),
                            ocr_confidence=round(float(ocr_conf), 4),
                            frame_number=frame_num,
                            video_timestamp=video_ts,
                            timestamp=datetime.now().isoformat(),
                            camera_id=self.camera_id,
                            latitude=cam_lat,
                            longitude=cam_lon,
                            location_source=cam_loc_src,
                            location_accuracy=cam_loc_acc,
                            plate_image_url=f"/detections/plates/{plate_filename}",
                            frame_image_url=f"/detections/frames/{frame_filename}",
                            is_target_match=False
                        )
                        self.all_detections.append(event)
                        self._dispatch_async(ws_manager.broadcast(event.model_dump()))
            else:
                with self.detections_lock:
                    if (time.time() - self.latest_detections_time) > 0.8:
                        self.latest_detections = []

            # Update latest annotated frame for live camera streaming endpoint
            self.camera.set_annotated_frame(self.get_display_frame(frame))

            # Small yield for thread switching
            time.sleep(0.005)

    def get_latest_vehicle(self, plate: str) -> Optional[DetectionEvent]:
        """Return the most recent detection for a specific plate."""
        norm_p = normalize_plate_text(plate)
        records = self.vehicle_records.get(norm_p, [])
        return records[-1] if records else None

    def get_vehicle_trajectory(self, plate: str) -> List[DetectionEvent]:
        """Return chronological detection timeline for a plate."""
        norm_p = normalize_plate_text(plate)
        return self.vehicle_records.get(norm_p, [])

    def get_detection_by_id(self, detection_id: str) -> Optional[DetectionEvent]:
        """Find a single detection by det_XXXXXX ID."""
        for d in self.all_detections:
            if d.detection_id == detection_id:
                return d
        return None
