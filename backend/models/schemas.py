from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AuthLoginRequest(BaseModel):
    username: str
    password: str

class AuthRegisterRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str = "operator"

class UserSession(BaseModel):
    user_id: Optional[int] = None
    username: str
    role: str
    authenticated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class SearchHistoryItem(BaseModel):
    id: int
    user_id: int
    plate_text: str
    state: str
    status: str
    created_at: str
    detection_count: int = 0

class CameraLocationRequest(BaseModel):
    camera_id: Optional[str] = "PHONE_CAM_01"
    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None
    source: Optional[str] = "gps"  # "gps" or "registered"

class CameraLocationResponse(BaseModel):
    camera_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str = "unknown"  # "gps", "registered", "ip", "unknown"
    location_accuracy: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DetectionEvent(BaseModel):
    type: str = "detection"
    detection_id: str
    track_id: Optional[int] = None
    plate_track_id: Optional[int] = None
    plate_text: Optional[str] = None
    target_plate: Optional[str] = None
    match_score: Optional[float] = None
    plate_confidence: float
    ocr_confidence: float
    frame_number: int
    video_timestamp: Optional[str] = None
    timestamp: str
    camera_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str = "unknown"  # "gps", "registered", "ip", "unknown"
    location_accuracy: Optional[float] = None
    plate_image_url: str
    frame_image_url: str
    is_target_match: bool = False

class TargetDetectionEvent(BaseModel):
    type: str = "target_detection"
    detection_id: str
    plate_text: str
    target_plate: str
    match_score: float
    plate_confidence: float
    ocr_confidence: float
    plate_image_url: str
    frame_image_url: str
    camera_id: str
    timestamp: str
    is_target_match: bool = True
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str = "unknown"
    location_accuracy: Optional[float] = None

class PipelineStateEvent(BaseModel):
    type: str = "pipeline_state"
    state: str  # camera_connecting, camera_connected, inference_running, plate_detected, ocr_processing, plate_match, no_match, camera_disconnected, video_finished, error
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None

class CameraMetadata(BaseModel):
    camera_id: str = "PHONE_CAM_01"
    name: str = "Test Traffic Camera"
    source: str = "video.mp4"
    status: str = "idle"  # idle, connecting, connected, disconnected, error
    fps: float = 0.0
    frame_count: int = 0
    current_frame: int = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_source: str = "unknown"
    location_accuracy: Optional[float] = None

class SearchRequest(BaseModel):
    plate: Optional[str] = None
    plate_number: Optional[str] = None
    state: Optional[str] = "Delhi"

    @property
    def target_plate_val(self) -> str:
        return self.plate or self.plate_number or "DL01AB1234"

class SearchResponse(BaseModel):
    session_id: str
    target_plate: str
    normalized_plate: str
    state: Optional[str] = "Delhi"
    status: str = "searching"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class CameraConfigRequest(BaseModel):
    source: Optional[str] = None
    url: Optional[str] = None
    camera_id: Optional[str] = None
    name: Optional[str] = None

    @property
    def source_val(self) -> str:
        return (self.source or self.url or "").strip()

class VehicleTrajectory(BaseModel):
    plate_text: str
    total_detections: int
    last_seen: Optional[DetectionEvent] = None
    detections: List[DetectionEvent] = []
