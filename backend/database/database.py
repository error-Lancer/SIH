"""
Phoenix ANPR — SQLite Database Layer
Handles persistent storage for users, target detections, and vehicle trajectories.
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import bcrypt

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "phoenix.db")


def get_db_path() -> str:
    """Return absolute path to SQLite database."""
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a connection with row_factory set to Row for dictionary-like access."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def init_db():
    """
    Initialize SQLite database and create required tables and indexes.
    Logs standard startup indicators:
    [DB] Initializing SQLite
    [DB] Database ready
    """
    print("[DB] Initializing SQLite", flush=True)
    logger.info("[DB] Initializing SQLite at %s", DB_PATH)
    os.makedirs(DATA_DIR, exist_ok=True)

    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # 2. Detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                target_plate TEXT,
                match_score REAL,
                plate_confidence REAL,
                ocr_confidence REAL,
                camera_id TEXT,
                timestamp TEXT NOT NULL,
                video_timestamp TEXT,
                latitude REAL,
                longitude REAL,
                location_source TEXT,
                location_accuracy REAL,
                plate_image_path TEXT,
                frame_image_path TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # 3. Search History table (user-specific queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                plate_text TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'Delhi',
                status TEXT NOT NULL DEFAULT 'SEARCHING',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # 4. Indexes for fast search
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_plate_text ON detections(plate_text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_camera_id ON detections(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_history_plate ON search_history(plate_text)")
        conn.commit()

        # Seed initial admin user if explicitly provided in environment
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        if user_count == 0:
            admin_user = os.getenv("ADMIN_USERNAME", "").strip()
            admin_pass = os.getenv("ADMIN_PASSWORD", "").strip()
            if admin_user and admin_pass:
                pw_hash = hash_password(admin_pass)
                now_iso = datetime.utcnow().isoformat()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (admin_user, pw_hash, now_iso)
                )
                conn.commit()
                logger.info("[DB] Seeded initial admin user: %s", admin_user)
            else:
                logger.warning("[DB] Skipping admin seeding: ADMIN_USERNAME or ADMIN_PASSWORD not set in environment")

    print("[DB] Database ready", flush=True)
    logger.info("[DB] Database ready")


# ==========================================
# User Management Operations
# ==========================================

def create_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Create a new user with bcrypt-hashed password.
    Returns sanitized user dict, or None if username already exists.
    """
    username = username.strip()
    if not username or not password:
        return None

    pw_hash = hash_password(password)
    now_iso = datetime.utcnow().isoformat()

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, now_iso)
            )
            conn.commit()
            user_id = cursor.lastrowid
            return {
                "id": user_id,
                "username": username,
                "created_at": now_iso
            }
    except sqlite3.IntegrityError:
        # Duplicate username
        return None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record including password hash (for authentication)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify user credentials against database hash.
    Returns sanitized user dict (without password_hash) if valid, else None.
    """
    user = get_user_by_username(username)
    if not user:
        return None
    if verify_password(password, user["password_hash"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"]
        }
    return None


# ==========================================
# Detection Storage Operations
# ==========================================

def insert_detection(data: Dict[str, Any]) -> int:
    """
    Insert a target match detection into SQLite.
    Returns the newly inserted row ID.
    """
    now_iso = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detections (
                plate_text, target_plate, match_score,
                plate_confidence, ocr_confidence, camera_id,
                timestamp, video_timestamp, latitude, longitude,
                location_source, location_accuracy,
                plate_image_path, frame_image_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("plate_text", ""),
            data.get("target_plate"),
            float(data.get("match_score", 1.0)) if data.get("match_score") is not None else None,
            float(data.get("plate_confidence", 0.0)) if data.get("plate_confidence") is not None else None,
            float(data.get("ocr_confidence", 0.0)) if data.get("ocr_confidence") is not None else None,
            data.get("camera_id", "PHONE_CAM_01"),
            data.get("timestamp", now_iso),
            data.get("video_timestamp"),
            float(data.get("latitude")) if data.get("latitude") is not None else None,
            float(data.get("longitude")) if data.get("longitude") is not None else None,
            data.get("location_source", "unknown"),
            float(data.get("location_accuracy")) if data.get("location_accuracy") is not None else None,
            data.get("plate_image_path", ""),
            data.get("frame_image_path", ""),
            now_iso
        ))
        conn.commit()
        return cursor.lastrowid


def get_detections_by_plate(plate_text: str) -> List[Dict[str, Any]]:
    """
    Retrieve all detections for a given plate, ordered chronologically.
    """
    clean_plate = "".join(c for c in plate_text.upper() if c.isalnum())
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM detections
            WHERE plate_text = ? OR target_plate = ?
            ORDER BY timestamp ASC, id ASC
        """, (clean_plate, clean_plate))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_trajectory_by_plate(plate_text: str) -> List[Dict[str, Any]]:
    """
    Retrieve chronological trajectory points for a given plate.
    Fields returned: plate, timestamp, camera_id, latitude, longitude,
    confidence, evidence images, location_source, location_accuracy.
    Coordinates are strictly null if not present (never fabricated).
    """
    clean_plate = "".join(c for c in plate_text.upper() if c.isalnum())
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                id,
                plate_text AS plate,
                timestamp,
                camera_id,
                latitude,
                longitude,
                location_source,
                location_accuracy,
                plate_confidence,
                ocr_confidence,
                match_score,
                plate_image_path,
                frame_image_path
            FROM detections
            WHERE plate_text = ? OR target_plate = ?
            ORDER BY timestamp ASC, id ASC
        """, (clean_plate, clean_plate))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


# ==========================================
# User-Specific Search History Operations
# ==========================================

def get_user_id_by_username(username: str) -> Optional[int]:
    """Retrieve database primary key ID for a username."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return row[0] if row else None


def insert_search_history(user_id: int, plate_text: str, state: str = "Delhi", status: str = "SEARCHING") -> int:
    """Record a user's search query in SQLite."""
    clean_plate = "".join(c for c in plate_text.upper() if c.isalnum())
    now_iso = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (user_id, plate_text, state, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, clean_plate, state, status, now_iso))
        conn.commit()
        return cursor.lastrowid


def get_user_search_history(user_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve search history belonging strictly to user_id.
    Includes real detection count from detections table.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.id,
                s.user_id,
                s.plate_text,
                s.state,
                s.status,
                s.created_at,
                (SELECT COUNT(*) FROM detections d WHERE d.plate_text = s.plate_text OR d.target_plate = s.plate_text) AS detection_count
            FROM search_history s
            WHERE s.user_id = ?
            ORDER BY s.id DESC
        """, (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_user_search_by_id(user_id: int, query_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve single search record belonging strictly to user_id.
    Enforces user isolation: User A cannot retrieve User B's search.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.id,
                s.user_id,
                s.plate_text,
                s.state,
                s.status,
                s.created_at,
                (SELECT COUNT(*) FROM detections d WHERE d.plate_text = s.plate_text OR d.target_plate = s.plate_text) AS detection_count
            FROM search_history s
            WHERE s.id = ? AND s.user_id = ?
        """, (query_id, user_id))
        row = cursor.fetchone()
        return dict(row) if row else None
