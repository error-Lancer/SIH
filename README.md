# 🔥 Phoenix ANPR
### AI-Powered Optical License-Plate Intelligence & Spatio-Temporal Tracking System

> **Phoenix ANPR** is an end-to-end, hardware-accelerated Automatic Number Plate Recognition (ANPR) system designed to detect, recognize, track, and geographically visualize vehicle license plates from live and recorded video sources.

---

## 🚀 Overview

Phoenix ANPR combines computer vision, OCR, real-time video processing, geospatial telemetry, and persistent storage into a single intelligence platform.

The system can ingest video from:

- 📱 Smartphone IP cameras
- 📹 RTSP security/CCTV feeds
- 🎞️ Recorded MP4/video files

The incoming video is processed through a real-time AI pipeline:

```text
Video Source
     │
     ▼
OpenCV Video Capture
     │
     ▼
YOLOv8 License-Plate Detection
     │
     ▼
Persistent Track ID
     │
     ▼
Plate Crop & Image Preprocessing
     │
     ▼
GPU-Accelerated EasyOCR
     │
     ▼
Plate Text + Confidence
     │
     ▼
Target Plate Matching
     │
     ▼
Telemetry + Evidence Storage
     │
     ▼
SQLite Database
     │
     ▼
FastAPI Backend
     │
     ├──────────────► WebSocket Live Events
     │
     ▼
Phoenix Tactical Dashboard
     │
     ├── Live Camera Feed
     ├── Plate Evidence
     ├── Detection History
     └── GIS Trajectory
