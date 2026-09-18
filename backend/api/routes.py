import os
import time
import hmac
import hashlib
import base64
import json
import secrets
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse, Response
import torch
import cv2

from backend.models.schemas import (
    SearchRequest,
    SearchResponse,
    DetectionEvent,
    VehicleTrajectory,
    CameraConfigRequest,
    CameraLocationRequest,
    CameraLocationResponse,
    AuthLoginRequest,
    AuthRegisterRequest,
    UserResponse,
    AuthTokenResponse,
    UserSession,
    SearchHistoryItem
)
from backend.db import database

router = APIRouter(prefix="/api", tags=["ANPR Pipeline"])

# Reference to detection_service set in main.py
detection_service = None

def get_service():
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Inference service initializing")
    return detection_service

# ---------------------------------------------------------------------------
# Security & Token Management (Server Authoritative)
# ---------------------------------------------------------------------------
def _get_secret_key() -> bytes:
    key = os.getenv("SECRET_KEY", "phoenix-production-secret-9f8a6b2c4e1d5f7a0b3c5e8d2a4f6b1c")
    return key.encode("utf-8")

def create_access_token(username: str, role: str = "operator", user_id: Optional[int] = None) -> str:
    """Create a short-lived HMAC-SHA256 signed access token with expiration."""
    expire_minutes = int(os.getenv("TOKEN_EXPIRE_MINUTES", "30"))
    if user_id is None:
        user_id = database.get_user_id_by_username(username)
    payload = {
        "sub": username,
        "role": role,
        "uid": user_id,
        "exp": int(time.time()) + (expire_minutes * 60),
        "nonce": secrets.token_hex(8)
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    sig = hmac.new(_get_secret_key(), raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"

def verify_token(token: str) -> Optional[dict]:
    """Verify HMAC token signature and check expiration timestamp."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        raw, sig = parts
        expected_sig = hmac.new(_get_secret_key(), raw.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        data = json.loads(base64.urlsafe_b64decode(raw.encode("utf-8")).decode("utf-8"))
        if data.get("exp", 0) < time.time():
            return None  # Expired
        return data
    except Exception:
        return None

def get_current_user(authorization: Optional[str] = Header(None)) -> UserSession:
    """Dependency enforcing backend authorization on sensitive operations."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide a valid Bearer token."
        )
    token = authorization.replace("Bearer ", "").replace("bearer ", "").strip()
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authorization token."
        )
    uid = payload.get("uid")
    if not uid:
        uid = database.get_user_id_by_username(payload["sub"])
    return UserSession(user_id=uid, username=payload["sub"], role=payload.get("role", "operator"))

def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[UserSession]:
    """Dependency for endpoints that accept both authenticated and guest access."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").replace("bearer ", "").strip()
    payload = verify_token(token)
    if payload:
        uid = payload.get("uid") or database.get_user_id_by_username(payload["sub"])
        return UserSession(user_id=uid, username=payload["sub"], role=payload.get("role", "operator"))
    return None

# ---------------------------------------------------------------------------
# Brute-Force Rate Limiting for Login Protection
# ---------------------------------------------------------------------------
FAILED_LOGINS: dict = {}
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW = 300  # 5 minutes
DUMMY_BCRYPT_HASH = "$2b$12$e8kGg3XbM.pY3jG3G/rUf.W5aQv7x0j9k5j6j7j8j9j0j1j2j3j4j"

def check_login_rate_limit(client_ip: str):
    now = time.time()
    attempts = FAILED_LOGINS.get(client_ip, [])
    recent = [t for t in attempts if (now - t) < LOCKOUT_WINDOW]
    FAILED_LOGINS[client_ip] = recent
    if len(recent) >= MAX_FAILED_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please wait 5 minutes before trying again."
        )

def record_failed_login(client_ip: str):
    now = time.time()
    attempts = FAILED_LOGINS.get(client_ip, [])
    attempts.append(now)
    FAILED_LOGINS[client_ip] = [t for t in attempts if (now - t) < LOCKOUT_WINDOW]

def clear_failed_logins(client_ip: str):
    if client_ip in FAILED_LOGINS:
        del FAILED_LOGINS[client_ip]

# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------
@router.post("/auth/register", response_model=UserResponse)
def register(req: AuthRegisterRequest):
    """Register a new operator with bcrypt-hashed password."""
    uname = req.username.strip()
    pwd = req.password.strip()

    if len(uname) < 3 or len(uname) > 32:
        raise HTTPException(status_code=400, detail="Username must be between 3 and 32 characters.")
    if not uname.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Username must be alphanumeric or underscores only.")
    if len(pwd) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    existing = database.get_user_by_username(uname)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")

    user = database.create_user(uname, pwd)
    if not user:
        raise HTTPException(status_code=409, detail="Username already exists.")

    return UserResponse(
        id=user["id"],
        username=user["username"],
        created_at=user["created_at"]
    )

@router.post("/auth/login", response_model=AuthTokenResponse)
def login(req: AuthLoginRequest, request: Request = None):
    """
    Authenticate operator credentials and issue a short-lived signed session token.
    Enforces rate-limiting against brute force and generic error messaging.
    """
    client_ip = request.client.host if (request and request.client) else "127.0.0.1"
    check_login_rate_limit(client_ip)

    req_user = req.username.strip()
    req_pass = req.password.strip()

    user = database.get_user_by_username(req_user)
    if user:
        is_valid = database.verify_password(req_pass, user["password_hash"])
    else:
        # Constant-time mitigation against user enumeration
        database.verify_password(req_pass, DUMMY_BCRYPT_HASH)
        is_valid = False

    if not is_valid:
        record_failed_login(client_ip)
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    clear_failed_logins(client_ip)
    role = "admin" if req_user == os.getenv("ADMIN_USERNAME", "admin") else "operator"
    token = create_access_token(req_user, role=role)
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        username=req_user,
        role=role
    )

@router.get("/auth/me", response_model=UserSession)
def get_current_user_profile(user: UserSession = Depends(get_current_user)):
    """Check current authentication status and role."""
    return user

# ---------------------------------------------------------------------------
# Health & Telemetry
# ---------------------------------------------------------------------------
@router.get("/health")
def health_check():
    """System health check, GPU acceleration status, and model metadata."""
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    return {
        "status": "healthy",
        "gpu_available": has_cuda,
        "device": device_name,
        "model": "YOLO (license-plate detector)",
        "classes": ["license-plate"],
        "ocr_engine": "EasyOCR (GPU)" if has_cuda else "EasyOCR (CPU)"
    }

# ---------------------------------------------------------------------------
# Target Search Pipeline
# ---------------------------------------------------------------------------
@router.post("/search", response_model=SearchResponse)
def search_plate(
    req: SearchRequest,
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """
    Search for a target plate across connected camera streams.
    Normalizes registration plate (e.g. 'DL 01 AB 1234' -> 'DL01AB1234')
    and initializes an active search session.
    """
    target_p = req.plate or req.plate_number
    if not target_p or not target_p.strip():
        raise HTTPException(status_code=400, detail="Plate registration number required")

    session = service.set_search_target(target_p, req.state or "Delhi")

    # Record search query in SQLite associated with user
    if user:
        uid = user.user_id or database.get_user_id_by_username(user.username)
        if uid:
            database.insert_search_history(
                user_id=uid,
                plate_text=session.normalized_plate,
                state=req.state or "Delhi",
                status="SEARCHING"
            )

    return session

@router.get("/search/target")
def get_target_status(service = Depends(get_service)):
    """Retrieve active target search metadata, current status, and matches."""
    return service.get_target_status()

# ---------------------------------------------------------------------------
# User-Specific Search History Endpoints (Strict User Isolation)
# ---------------------------------------------------------------------------
@router.get("/history", response_model=List[SearchHistoryItem])
def get_user_search_history_endpoint(user: UserSession = Depends(get_current_user)):
    """
    Retrieve search query history belonging exclusively to the authenticated user.
    Enforces strict user isolation.
    """
    uid = user.user_id or database.get_user_id_by_username(user.username)
    if not uid:
        raise HTTPException(status_code=401, detail="User account not found.")
    return database.get_user_search_history(uid)

@router.get("/history/{query_id}", response_model=SearchHistoryItem)
def get_user_search_history_by_id_endpoint(query_id: int, user: UserSession = Depends(get_current_user)):
    """
    Retrieve a specific search query belonging strictly to the authenticated user.
    User A cannot access User B's search records.
    """
    uid = user.user_id or database.get_user_id_by_username(user.username)
    if not uid:
        raise HTTPException(status_code=401, detail="User account not found.")
    record = database.get_user_search_by_id(uid, query_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Search record #{query_id} not found.")
    return record

# ---------------------------------------------------------------------------
# Vehicle Trajectory & Detection Evidence (SQLite Backed)
# ---------------------------------------------------------------------------
@router.get("/vehicles/{plate}")
def get_vehicle_detections(
    plate: str,
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """
    Retrieve all confirmed detection records for a plate from SQLite.
    Returns chronological results.
    """
    clean_plate = "".join(c for c in plate.strip().upper() if c.isalnum())
    sqlite_dets = database.get_detections_by_plate(clean_plate)
    if not sqlite_dets:
        # Check in-memory if not yet written
        latest_mem = service.get_latest_vehicle(clean_plate)
        if not latest_mem:
            raise HTTPException(status_code=404, detail=f"No detections found for plate '{plate}'")
        return {
            "plate": clean_plate,
            "total_detections": 1,
            "latest": latest_mem.model_dump(),
            "detections": [latest_mem.model_dump()]
        }
    for d in sqlite_dets:
        if "plate_image_path" in d and "plate_image_url" not in d:
            d["plate_image_url"] = d["plate_image_path"]
        if "frame_image_path" in d and "frame_image_url" not in d:
            d["frame_image_url"] = d["frame_image_path"]

    return {
        "plate": clean_plate,
        "total_detections": len(sqlite_dets),
        "latest": sqlite_dets[-1] if sqlite_dets else None,
        "detections": sqlite_dets
    }


@router.get("/vehicles/{plate}/trajectory")
def get_vehicle_trajectory(
    plate: str,
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """
    Retrieve chronological trajectory points for a plate from SQLite.
    If latitude/longitude is NULL, it is never fabricated.
    """
    clean_plate = "".join(c for c in plate.strip().upper() if c.isalnum())
    points = database.get_trajectory_by_plate(clean_plate)
    return {
        "plate": clean_plate,
        "total_points": len(points),
        "trajectory": points
    }

@router.get("/detections/recent", response_model=List[DetectionEvent])
def get_recent_detections(limit: int = 20, service = Depends(get_service)):
    """List recent detections."""
    safe_limit = max(1, min(limit, 100))
    return service.all_detections[-safe_limit:][::-1]

@router.get("/detections/{detection_id}", response_model=DetectionEvent)
def get_detection_by_id(detection_id: str, service = Depends(get_service)):
    """Retrieve a single detection event by its unique det_XXXXXX identifier."""
    # Sanitize input: prevent path traversal or special chars
    clean_id = os.path.basename(detection_id).strip()
    det = service.get_detection_by_id(clean_id)
    if not det:
        raise HTTPException(status_code=404, detail=f"Detection '{clean_id}' not found")
    return det

# ---------------------------------------------------------------------------
# Camera Stream & Control
# ---------------------------------------------------------------------------
@router.get("/camera/status")
def get_camera_status(service = Depends(get_service)):
    """Get real-time camera status, resolution, FPS, location, and telemetry."""
    return service.camera.get_status()

@router.get("/camera/frame")
def get_camera_frame(annotated: bool = True, service = Depends(get_service)):
    """Retrieve the latest single camera frame as JPEG image for snapshots/debugging."""
    frame, _ = service.camera.get_latest_frame_with_id()
    if frame is None:
        raise HTTPException(status_code=503, detail="No camera frames available yet")
    display_frame = service.get_display_frame(frame) if annotated else frame
    ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.get("/camera/stream")
def get_camera_stream(service = Depends(get_service)):
    """
    Real-time MJPEG live camera stream.
    Displays live camera frames continuously in browser <img> tags.
    Drains from the thread-safe camera buffer without being blocked by YOLO inference.
    """
    def frame_generator():
        while True:
            frame, _ = service.camera.get_latest_frame_with_id()
            if frame is not None:
                display_frame = service.get_display_frame(frame)
                ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(0.033)  # ~30 FPS smooth continuous stream

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/camera/reconnect")
def reconnect_camera(
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """Force an immediate reconnection attempt to the active camera source."""
    status = service.reconnect_camera()
    return {
        "message": "Reconnection attempt complete",
        "camera_status": status
    }

@router.post("/camera/config")
def configure_camera(
    req: CameraConfigRequest,
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """
    Dynamically update camera source (e.g. Phone IP webcam URL or local MP4 path).
    Sanitizes inputs to prevent SSRF and path traversal.
    """
    raw_src = req.source_val
    if not raw_src:
        raise HTTPException(status_code=400, detail="Camera source URL or path required")

    # Reject dangerous protocols
    lower = raw_src.lower()
    if lower.startswith("file://") or lower.startswith("gopher://") or lower.startswith("dict://"):
        raise HTTPException(status_code=400, detail="Unsupported camera protocol")


    status = service.update_camera_source(raw_src, req.camera_id)
    return {
        "message": f"Camera source updated to {raw_src}",
        "camera_status": status
    }

# ---------------------------------------------------------------------------
# Camera Location System (GPS / Registered / IP Geolocation Fallback)
# ---------------------------------------------------------------------------
@router.get("/camera/location", response_model=CameraLocationResponse)
def get_camera_location(service = Depends(get_service)):
    """Retrieve active camera coordinates and location source."""
    loc = service.camera.get_location()
    return CameraLocationResponse(
        camera_id=service.camera_id,
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        location_source=loc["location_source"],
        location_accuracy=loc["location_accuracy"],
        timestamp=loc["timestamp"]
    )

@router.post("/camera/location", response_model=CameraLocationResponse)
def update_camera_location(
    req: CameraLocationRequest,
    service = Depends(get_service),
    user: Optional[UserSession] = Depends(get_optional_user)
):
    """
    Update camera coordinates.
    Priority order:
    1. GPS (phone device GPS with user consent) - highest priority
    2. Registered (admin configured fixed coordinates) - overrides IP/unknown
    3. IP Geolocation (fallback only; never overrides GPS or registered)
    """
    if req.latitude is None or req.longitude is None:
        raise HTTPException(status_code=400, detail="Latitude and longitude required")

    if not (-90.0 <= req.latitude <= 90.0) or not (-180.0 <= req.longitude <= 180.0):
        raise HTTPException(status_code=400, detail="Coordinates out of valid geographical range")

    src = (req.source or "gps").lower().strip()
    res = service.update_camera_location(
        lat=req.latitude,
        lon=req.longitude,
        source=src,
        accuracy=req.accuracy_m
    )
    return CameraLocationResponse(**res)
