import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

class ANPRDetector:
    """
    License plate detector wrapping the existing YOLO model ('best.pt').
    NOTE: This model is strictly a license-plate detector (class 0: license-plate),
    NOT a vehicle detector.
    """
    def __init__(self, weights_path: str = "best.pt", conf_threshold: float = 0.25):
        if not os.path.exists(weights_path):
            # Try looking relative to project root
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), weights_path)
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                raise FileNotFoundError(f"YOLO weights not found at '{weights_path}' or '{alt_path}'")

        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[ANPRDetector] Loading YOLO license-plate model from '{weights_path}' onto {self.device.upper()}...")
        self.model = YOLO(weights_path)
        self.classes = self.model.names  # {0: 'license-plate'}
        print(f"[ANPRDetector] Model ready. Detected classes: {self.classes}")

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Run inference on a single image/frame.
        Returns a list of detected license plate dictionaries.
        """
        if frame is None or frame.size == 0:
            return []

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        with torch.no_grad():
            results = self.model.predict(
                frame,
                conf=conf,
                device=self.device,
                verbose=False,
                imgsz=640
            )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        h, w = frame.shape[:2]

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            confidence = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.classes.get(cls_id, "license-plate")

            # Clamp coordinates to frame boundary
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 4),
                "class_id": cls_id,
                "class_name": cls_name
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw high-tech HUD-style bounding boxes on the full frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            plate_text = det.get("plate_text")
            track_id = det.get("track_id")

            # Box color: Vibrant Flame Orange / Red
            color = (55, 86, 255)  # BGR: #ff5637
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Reticle corner accents
            corner_len = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
            corner_color = (0, 107, 254)  # BGR: #fe6b00
            if corner_len > 2:
                # Top-left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), corner_color, 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), corner_color, 3)
                # Top-right
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), corner_color, 3)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), corner_color, 3)
                # Bottom-left
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), corner_color, 3)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), corner_color, 3)
                # Bottom-right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), corner_color, 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), corner_color, 3)

            # Label banner
            label_parts = ["PLATE"]
            if track_id is not None:
                label_parts.append(f"#{track_id}")
            if plate_text:
                label_parts.append(f"[{plate_text}]")
            label_parts.append(f"{int(conf * 100)}%")
            label = " ".join(label_parts)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_y1 = max(0, y1 - th - 6)
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 8, bg_y1 + th + 6), (19, 19, 23), -1)
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 8, bg_y1 + th + 6), color, 1)
            cv2.putText(annotated, label, (x1 + 4, bg_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 225, 228), 1, cv2.LINE_AA)

        return annotated
