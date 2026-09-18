import re
import cv2
import numpy as np
from typing import Tuple, Optional
import easyocr

import difflib

def normalize_plate_text(text: str) -> str:
    """
    Strip spaces, punctuation, and convert to uppercase.
    Also handles standard Indian HSRP 'IND' prefix watermark.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    # Strip leading 'IND' country code watermark if present and remaining plate is at least 6 characters
    if cleaned.startswith("IND") and len(cleaned) >= 7:
        cleaned = cleaned[3:]
    return cleaned

def calculate_plate_similarity(target: str, detected: str) -> float:
    """
    Calculate fuzzy similarity score between target plate and OCR detected text (0.0 to 1.0).
    Handles minor OCR errors (e.g. 0/O, 1/I, 8/B substitutions, single-char edit distance).
    """
    norm_target = normalize_plate_text(target)
    norm_detected = normalize_plate_text(detected)

    if not norm_target or not norm_detected:
        return 0.0

    # Exact normalized match
    if norm_target == norm_detected:
        return 1.0

    # Direct canonical confusion substitution check (common optical OCR ambiguities)
    def canonical_confusion(s: str) -> str:
        return (
            s.replace("O", "0")
             .replace("I", "1")
             .replace("Z", "2")
             .replace("B", "8")
             .replace("S", "5")
        )

    if canonical_confusion(norm_target) == canonical_confusion(norm_detected):
        return 0.98

    # Sequence similarity / edit distance ratio
    ratio = difflib.SequenceMatcher(None, norm_target, norm_detected).ratio()
    return round(ratio, 4)

class PlateOCR:
    """
    License plate character recognition engine wrapping EasyOCR.
    Applies image preprocessing tailored for vehicle number plates.
    Ensures zero hallucination: returns (None, 0.0) if plate text is unreadable or uncertain.
    """
    def __init__(self, use_gpu: bool = True, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        print(f"[PlateOCR] Initializing EasyOCR reader (GPU={use_gpu})...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        print("[PlateOCR] EasyOCR ready.")

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate crop for optimal optical character recognition:
        1. Rescale if too small.
        2. Convert to grayscale.
        3. Bilateral filter for noise removal while keeping character edges sharp.
        4. Contrast Limiting Adaptive Histogram Equalization (CLAHE).
        """
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        
        # Upscale small crops so character height is at least ~60-80px
        target_h = 100
        if h < target_h:
            scale = target_h / float(h)
            new_w = max(int(w * scale), 200)
            crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Noise reduction preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def read_plate(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Read text from cropped plate image.
        Returns:
            (plate_text, ocr_confidence) if successful and confident,
            or (None, 0.0) if uncertain.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        try:
            processed = self.preprocess(crop)

            # Try reading on both processed and original grayscale
            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            if not results:
                # Fallback on raw crop
                results = self.reader.readtext(
                    crop,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

            if not results:
                return None, 0.0

            # Combine recognized text segments sorted left-to-right
            # result item format: (bbox, text, conf)
            sorted_segments = sorted(results, key=lambda item: item[0][0][0] if len(item[0]) > 0 else 0)

            all_chars = []
            total_conf = 0.0
            valid_segments = 0

            for _, raw_text, conf in sorted_segments:
                cleaned = normalize_plate_text(raw_text)
                if cleaned:
                    all_chars.append(cleaned)
                    total_conf += float(conf)
                    valid_segments += 1

            if not all_chars or valid_segments == 0:
                return None, 0.0

            avg_conf = total_conf / valid_segments
            combined_text = "".join(all_chars)

            # Sanity check: plates typically have at least 4 alphanumeric chars
            if len(combined_text) < 4 or avg_conf < self.min_confidence:
                # Return null with 0.0 as requested when uncertain
                return None, 0.0

            return combined_text, round(avg_conf, 4)

        except Exception as e:
            print(f"[PlateOCR] Error during OCR processing: {e}")
            return None, 0.0
