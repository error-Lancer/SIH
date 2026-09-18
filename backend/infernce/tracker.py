from typing import List, Dict, Any, Optional

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class PlateTracker:
    """
    Tracks detected license plate bounding boxes across consecutive frames.
    Assigns a persistent 'plate_track_id' (or 'track_id') to each plate track.
    NOTE: Tracks license plates, NOT vehicles (since best.pt is a plate-only detector).
    """
    def __init__(self, iou_threshold: float = 0.35, max_missed_frames: int = 25):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match current frame detections to existing active tracks.
        Assigns 'track_id' / 'plate_track_id' to each detection.
        """
        # Increment missed frames for all active tracks
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid]["missed_frames"] += 1
            if self.active_tracks[tid]["missed_frames"] > self.max_missed_frames:
                del self.active_tracks[tid]

        if not detections:
            return []

        matched_detections = []
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        # Greedy IoU matching
        for tid, track_info in self.active_tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for d_idx in unmatched_dets:
                iou = compute_iou(track_info["bbox"], detections[d_idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou >= self.iou_threshold and best_det_idx != -1:
                det = detections[best_det_idx].copy()
                det["track_id"] = tid
                det["plate_track_id"] = tid
                matched_detections.append(det)
                matched_tracks.add(tid)
                unmatched_dets.remove(best_det_idx)

                # Update track state
                self.active_tracks[tid]["bbox"] = det["bbox"]
                self.active_tracks[tid]["missed_frames"] = 0
                self.active_tracks[tid]["hits"] += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            tid = self.next_track_id
            self.next_track_id += 1

            det = detections[d_idx].copy()
            det["track_id"] = tid
            det["plate_track_id"] = tid
            matched_detections.append(det)

            self.active_tracks[tid] = {
                "track_id": tid,
                "bbox": det["bbox"],
                "missed_frames": 0,
                "hits": 1
            }

        return matched_detections
