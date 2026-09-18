import os
import sys
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import routes
from backend.api.websocket import ws_manager
from backend.services.detection_service import DetectionService
from backend.db.database import init_db

# Global service instance
detection_service: DetectionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detection_service
    # Initialize SQLite Database
    init_db()

    # Read environment variables
    app_env = os.getenv("APP_ENV", "production").lower()
    weights_path = os.getenv("YOLO_WEIGHTS", os.path.join(root_dir, "best.pt"))
    camera_source = os.getenv("CAMERA_SOURCE", "video.mp4")
    camera_id = os.getenv("CAMERA_ID", "PHONE_CAM_01")
    cooldown = float(os.getenv("DETECTION_COOLDOWN_SECONDS", "2.0"))
    debug_dets = os.getenv("DEBUG_DETECTIONS", "false").lower() in ["true", "1", "yes"]

    print("=" * 60)
    print("🚀 INITIALIZING PHOENIX ANPR INFERENCE BACKEND")
    print(f"   Environment:      {app_env}")
    print(f"   YOLO Weights:     {weights_path}")
    print(f"   Camera Source:    {camera_source}")
    print(f"   Camera ID:        {camera_id}")
    print(f"   Cooldown Secs:    {cooldown}s")
    print(f"   Debug Detections: {debug_dets}")
    print("=" * 60)

    # Initialize DetectionService
    detection_service = DetectionService(
        weights_path=weights_path,
        camera_source=camera_source,
        camera_id=camera_id,
        cooldown_seconds=cooldown,
        debug_detections=debug_dets
    )
    routes.detection_service = detection_service

    # Attach FastAPI's event loop and start background inference
    loop = asyncio.get_running_loop()
    detection_service.set_event_loop(loop)
    detection_service.start()

    yield

    # Shutdown
    if detection_service:
        detection_service.stop()
    print("🛑 Phoenix ANPR Backend stopped.")

app = FastAPI(
    title="Phoenix ANPR — AI Vehicle Intelligence Backend",
    description="Real-time Optical Number Plate Recognition & Telemetry Pipeline",
    version="1.0.0",
    docs_url=None if os.getenv("APP_ENV", "production").lower() == "production" else "/docs",
    redoc_url=None,
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# Production CORS Configuration
# ---------------------------------------------------------------------------
cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500,null")
allowed_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
if not allowed_origins:
    allowed_origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Production Exception Handler (Prevents Stack Trace Leakage)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def production_exception_handler(request: Request, exc: Exception):
    app_env = os.getenv("APP_ENV", "production").lower()
    debug_mode = os.getenv("DEBUG", "false").lower() in ["true", "1", "yes"]
    if not debug_mode or app_env == "production":
        # In production or when DEBUG=false, return generic error without internal path traces
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred. Please contact system administrator."}
        )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

# ---------------------------------------------------------------------------
# Route Registration & Static Mounts
# ---------------------------------------------------------------------------
app.include_router(routes.router)

# Mount static directories for model-generated evidence
plates_dir = os.path.join(root_dir, "backend", "detections", "plates")
frames_dir = os.path.join(root_dir, "backend", "detections", "frames")
os.makedirs(plates_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

app.mount("/detections/plates", StaticFiles(directory=plates_dir), name="plates")
app.mount("/detections/frames", StaticFiles(directory=frames_dir), name="frames")

# Mount production UI
ui_dir = os.path.join(root_dir, "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
def root():
    """Redirect root to the Phoenix ANPR production UI."""
    return RedirectResponse(url="/ui/")

# ---------------------------------------------------------------------------
# WebSocket Endpoint for Real-Time Detection Events & Telemetry
# ---------------------------------------------------------------------------
@app.websocket("/ws/detections")
async def websocket_detections_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send initial camera status & location
        if detection_service and detection_service.camera:
            cam_stat = detection_service.camera.get_status()
            await websocket.send_json({
                "type": "camera_status",
                **cam_stat
            })

        while True:
            # Maintain active bidirectional connection
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
