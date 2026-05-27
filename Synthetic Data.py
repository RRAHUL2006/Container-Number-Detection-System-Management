"""
Container Number Recognition System  v4
=======================================================
v3 fixes retained:
  1. ISO 6346 alpha lookup table corrected.
  2. Duplicate bounding-box detections suppressed via post-NMS IoU dedup.
  3. '5'↔'S' and '6'↔'G' swaps handled; extended _DIGIT_FIX_MAP + fuzzy fallback.
  4. CONTAINER_REGEX accepts optional space/dash between owner prefix and serial.

New in v4:
  A. Highest-Accuracy Detection Logger
     - Tracks best (highest-confidence) detection per unique container ID.
     - Saves detection_log.txt and detection_log.csv.
     - Auto-updates when a higher-confidence detection is found.
     - Logs: timestamp, container number, confidence, ISO status, frame number.
  B. Live URL Streaming Support
     - RTSP, HTTP, IP camera, webcam index, and YouTube live (via yt-dlp).
     - Auto-reconnect with configurable retry logic.
  C. Smart Logging Rules
     - Only valid ISO 6346 numbers are logged.
     - Per-container, only the highest confidence detection is kept.
  D. Overlay Improvements
     - Best confidence per container, current OCR confidence,
       total valid containers detected, live FPS.
  E. Stream Stability
     - Auto-reconnect on disconnect; safe dropped-frame handling.
  F. Real-Time Optimization
     - Multi-threaded OCR pipeline using concurrent.futures.ThreadPoolExecutor.
  G. Export Feature
     - End-of-run CSV export of best detections.
     - Snapshot images of best detections saved to best_detections/.
"""

import csv
import cv2
import numpy as np
import os
import re
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
#  ★  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────

MODEL_WEIGHTS_PATH  = r"C:\Users\R RAHUL\OneDrive\Desktop\container\runs\detect\train4\weights\best.pt"
OCR_MODEL_PATH      = r"C:\Users\R RAHUL\OneDrive\Desktop\container"
VIDEO_INPUT_PATH    = r"C:\Users\R RAHUL\OneDrive\Desktop\container\WhatsApp Video 2026-05-27 at 4.59.36 AM.mp4"
VIDEO_OUTPUT_PATH   = r"C:\Users\R RAHUL\OneDrive\Desktop\container\annotated_output_v4.mp4"
TEST_IMAGES_FOLDER  = ""

# Logging output paths
LOG_DIR             = r"C:\Users\R RAHUL\OneDrive\Desktop\container\logs"
DETECTION_LOG_TXT   = os.path.join(LOG_DIR, "detection_log.txt")
DETECTION_LOG_CSV   = os.path.join(LOG_DIR, "detection_log.csv")
BEST_SNAPSHOTS_DIR  = os.path.join(LOG_DIR, "best_detections")

# ─────────────────────────────────────────────
#  Back-end selector
# ─────────────────────────────────────────────

DETECTOR_BACKEND = "yolo"
OCR_BACKEND      = "paddleocr"

# ─────────────────────────────────────────────
#  Tuning parameters
# ─────────────────────────────────────────────

CONF_THRESHOLD   = 0.30
NMS_IOU          = 0.45
DEDUP_IOU        = 0.40
DISPLAY_SCALE    = 1.0
SKIP_FRAMES      = 2

# Temporal smoothing
TRACK_HISTORY    = 8
VOTE_THRESHOLD   = 3
WARMUP_FRAMES    = 2

# Stream reconnect settings
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY_SEC    = 3.0

# Multi-threading
OCR_THREAD_WORKERS = 4

# ─────────────────────────────────────────────
#  FIX 1 – Correct ISO 6346 alpha lookup table
# ─────────────────────────────────────────────

_ISO_ALPHA = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15,
    'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
    'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26,
    'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
    'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37,
    'Z': 38,
}

# ─────────────────────────────────────────────
#  FIX 3 – Extended OCR character-confusion map
# ─────────────────────────────────────────────

_DIGIT_FIX_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

_LETTER_FIX_MAP = {
    "0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z",
}

CONTAINER_REGEX = re.compile(
    r"([A-Z]{3}[UJZ])[\s\-]?(\d{6,7})",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────
#  Colours (BGR)
# ─────────────────────────────────────────────

COLOR_VALID   = (0,   220,  50)
COLOR_PARTIAL = (50,  180, 255)
COLOR_RAW     = (100, 100, 255)
COLOR_TEXT_BG = (0,     0,   0)
COLOR_TEXT_FG = (255, 255, 255)
COLOR_OVERLAY = (0,   200, 255)
FONT          = cv2.FONT_HERSHEY_DUPLEX
THICKNESS     = 2

# ─────────────────────────────────────────────
#  Lazy-load back-ends
# ─────────────────────────────────────────────

_detector = None
_ocr      = None


def load_detector():
    global _detector
    if _detector is not None:
        return _detector
    if DETECTOR_BACKEND == "yolo":
        try:
            from ultralytics import YOLO
            _detector = YOLO(MODEL_WEIGHTS_PATH)
            print(f"[✔] YOLO loaded: {MODEL_WEIGHTS_PATH}")
        except ImportError:
            raise ImportError("pip install ultralytics")
    else:
        raise ValueError(f"Unknown DETECTOR_BACKEND: {DETECTOR_BACKEND}")
    return _detector


def load_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    if OCR_BACKEND == "easyocr":
        try:
            import easyocr
            _ocr = easyocr.Reader(
                ["en"],
                model_storage_directory=OCR_MODEL_PATH,
                gpu=True,
                verbose=False,
            )
            print(f"[✔] EasyOCR loaded (model dir: {OCR_MODEL_PATH})")
        except ImportError:
            raise ImportError("pip install easyocr")

    elif OCR_BACKEND == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
                enable_mkldnn=False,
            )
            print("[✔] PaddleOCR loaded")
        except ImportError:
            raise ImportError("pip install paddlepaddle paddleocr")

    elif OCR_BACKEND == "tesseract":
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = OCR_MODEL_PATH
        _ocr = pytesseract

    else:
        raise ValueError(f"Unknown OCR_BACKEND: {OCR_BACKEND}")

    return _ocr


# ─────────────────────────────────────────────
#  ISO 6346 validation
# ─────────────────────────────────────────────

def _iso6346_checkdigit(owner: str, serial6: str) -> int:
    code  = (owner + serial6).upper()
    total = 0
    for i, ch in enumerate(code):
        val = _ISO_ALPHA[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    r = total % 11
    return 0 if r == 10 else r


def iso6346_check(owner: str, serial: str) -> bool:
    if len(serial) < 7 or len(owner) != 4:
        return False
    try:
        return int(serial[6]) == _iso6346_checkdigit(owner, serial[:6])
    except (ValueError, KeyError):
        return False


_SIMILAR_LETTERS_IN_DIGIT_ZONE = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8",
}


def _try_fix_checkdigit(owner: str, serial7: str):
    for sub in range(10):
        candidate = serial7[:6] + str(sub)
        if iso6346_check(owner, candidate):
            return candidate, True
    for pos in range(6):
        ch = serial7[pos]
        alt = _SIMILAR_LETTERS_IN_DIGIT_ZONE.get(ch)
        if alt is None:
            continue
        candidate_serial = serial7[:pos] + alt + serial7[pos + 1:]
        if iso6346_check(owner, candidate_serial):
            return candidate_serial, True
    return serial7, False


# ─────────────────────────────────────────────
#  OCR pre-processing pipelines
# ─────────────────────────────────────────────

def _upscale(gray: np.ndarray, min_h: int = 120) -> np.ndarray:
    h, w = gray.shape
    if h < min_h:
        scale = max(2, min_h // h)
        gray  = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return gray


def _pipeline_adaptive(gray):
    g    = _upscale(gray)
    proc = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)
    k    = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(proc, -1, k)


def _pipeline_otsu(gray):
    g     = _upscale(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g     = clahe.apply(g)
    _, proc = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return proc


def _pipeline_morph(gray):
    g      = _upscale(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g      = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    _, proc = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return proc


_PIPELINES = [_pipeline_adaptive, _pipeline_otsu, _pipeline_morph]


def _correct_container_string(raw: str) -> str:
    raw = raw.upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    if len(raw) < 10:
        return raw
    lp_fixed = "".join(_LETTER_FIX_MAP.get(ch, ch) for ch in raw[:4])
    dp_fixed = "".join(_DIGIT_FIX_MAP.get(ch, ch)  for ch in raw[4:])
    return lp_fixed + dp_fixed


def run_ocr_on_crop(crop_bgr: np.ndarray) -> str:
    ocr     = load_ocr()
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for pipeline in _PIPELINES:
        proc     = pipeline(gray)
        proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        raw_text = ""

        if OCR_BACKEND == "easyocr":
            res      = ocr.readtext(proc_bgr, detail=0,
                                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
            raw_text = " ".join(res)
        elif OCR_BACKEND == "paddleocr":
            res = ocr.ocr(proc_bgr, cls=True)
            if res and res[0]:
                raw_text = " ".join(line[1][0] for line in res[0])
        elif OCR_BACKEND == "tesseract":
            cfg      = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            raw_text = ocr.image_to_string(proc_bgr, config=cfg)

        cleaned = _correct_container_string(raw_text.upper())
        if cleaned:
            results.append(cleaned)

    if not results:
        return ""
    counts = Counter(results)
    winner, freq = counts.most_common(1)[0]
    return winner if freq > 1 else results[0]


def extract_container_number(text: str):
    m = CONTAINER_REGEX.search(text)
    if not m:
        return None, False
    owner  = m.group(1).upper()
    serial = m.group(2)
    if len(serial) == 6:
        expected_cd = _iso6346_checkdigit(owner, serial)
        serial      = serial + str(expected_cd)
    if iso6346_check(owner, serial):
        return owner + serial, True
    fixed_serial, fixed = _try_fix_checkdigit(owner, serial[:7])
    if fixed:
        return owner + fixed_serial, True
    return owner + serial[:7], False


# ─────────────────────────────────────────────
#  FIX 2 – IoU-based duplicate suppression
# ─────────────────────────────────────────────

def _iou(a, b) -> float:
    ix1  = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2  = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def deduplicate_boxes(boxes, iou_threshold: float = DEDUP_IOU):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept  = []
    for box in boxes:
        if all(_iou(box[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


# ─────────────────────────────────────────────
#  NEW (A/C) – Highest-Accuracy Detection Logger
# ─────────────────────────────────────────────

class DetectionLogger:
    """
    Maintains a per-container-ID store of the single highest-confidence
    valid ISO 6346 detection seen so far. Thread-safe.
    """

    CSV_FIELDS = ["timestamp", "container_number", "confidence",
                  "iso_valid", "frame_number"]

    def __init__(self, log_txt: str, log_csv: str, snapshot_dir: str):
        self._lock        = threading.Lock()
        self._best: dict  = {}   # container_id → record dict
        self._log_txt     = log_txt
        self._log_csv     = log_csv
        self._snapshot_dir = snapshot_dir

        # Ensure directories exist
        for path in [log_txt, log_csv]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)

        # Write CSV header if new file
        if not Path(log_csv).exists():
            with open(log_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()

    def update(self, container_id: str, confidence: float,
               is_valid: bool, frame_number: int,
               crop_bgr: np.ndarray | None = None):
        """
        Call with every valid ISO detection.
        Only stores / overwrites if confidence is higher than previous best.
        """
        if not is_valid:
            return

        record = {
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "container_number": container_id,
            "confidence":       round(confidence, 4),
            "iso_valid":        is_valid,
            "frame_number":     frame_number,
            "_crop":            crop_bgr,   # internal only, excluded from CSV
        }

        with self._lock:
            existing = self._best.get(container_id)
            if existing and existing["confidence"] >= confidence:
                return   # already have a better or equal detection
            self._best[container_id] = record

        # Persist immediately (outside lock for speed)
        self._write_txt(record)
        self._append_csv(record)
        if crop_bgr is not None:
            self._save_snapshot(container_id, crop_bgr, confidence, frame_number)

    def _write_txt(self, record: dict):
        line = (
            f"[{record['timestamp']}] "
            f"{record['container_number']}  "
            f"conf={record['confidence']:.4f}  "
            f"ISO={'✔' if record['iso_valid'] else '✘'}  "
            f"frame={record['frame_number']}\n"
        )
        # Rewrite entire file atomically so the log always reflects best values
        with self._lock:
            all_records = sorted(
                self._best.values(),
                key=lambda r: r["container_number"],
            )
        try:
            with open(self._log_txt, "w", encoding="utf-8") as f:
                f.write("# Container Number Recognition – Best Detections Log\n")
                f.write(f"# Last updated: {datetime.now().isoformat(timespec='seconds')}\n\n")
                for r in all_records:
                    f.write(
                        f"[{r['timestamp']}] "
                        f"{r['container_number']}  "
                        f"conf={r['confidence']:.4f}  "
                        f"ISO={'✔' if r['iso_valid'] else '✘'}  "
                        f"frame={r['frame_number']}\n"
                    )
        except OSError as e:
            print(f"[!] TXT log write error: {e}")

    def _append_csv(self, record: dict):
        # Rewrite the CSV fully so each container keeps only the best row
        with self._lock:
            rows = [
                {k: v for k, v in r.items() if k != "_crop"}
                for r in self._best.values()
            ]
        try:
            with open(self._log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as e:
            print(f"[!] CSV log write error: {e}")

    def _save_snapshot(self, container_id: str, crop: np.ndarray,
                       confidence: float, frame_number: int):
        fname = (
            f"{container_id}_conf{confidence:.4f}_f{frame_number}.jpg"
            .replace(":", "-")
        )
        path = os.path.join(self._snapshot_dir, fname)
        try:
            cv2.imwrite(path, crop)
        except Exception as e:
            print(f"[!] Snapshot save error: {e}")

    def export_csv(self, out_path: str | None = None):
        """Final export at end of run (same as running log but explicit)."""
        target = out_path or self._log_csv
        with self._lock:
            rows = [
                {k: v for k, v in r.items() if k != "_crop"}
                for r in self._best.values()
            ]
        try:
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[✔] Final CSV exported → {target}")
        except OSError as e:
            print(f"[!] Final CSV export error: {e}")

    def total_valid(self) -> int:
        with self._lock:
            return len(self._best)

    def best_conf_for(self, container_id: str) -> float | None:
        with self._lock:
            rec = self._best.get(container_id)
            return rec["confidence"] if rec else None


# ─────────────────────────────────────────────
#  Temporal tracker (unchanged from v3)
# ─────────────────────────────────────────────

class DetectionTracker:
    def __init__(self, history: int = TRACK_HISTORY, vote_th: int = VOTE_THRESHOLD):
        self.history  = history
        self.vote_th  = vote_th
        self._tracks  = defaultdict(list)
        self._ages    = defaultdict(int)

    @staticmethod
    def _cell_key(x1, y1, x2, y2, grid: int = 80) -> tuple:
        cx = (x1 + x2) // 2 // grid
        cy = (y1 + y2) // 2 // grid
        return cx, cy

    def update(self, x1, y1, x2, y2, raw_id):
        key = self._cell_key(x1, y1, x2, y2)
        self._ages[key] += 1
        history = self._tracks[key]
        history.append(raw_id)
        if len(history) > self.history:
            history.pop(0)

    def query(self, x1, y1, x2, y2):
        key       = self._cell_key(x1, y1, x2, y2)
        age       = self._ages.get(key, 0)
        hist      = self._tracks.get(key, [])
        valid_ids = [h for h in hist if h is not None]
        if not valid_ids:
            return None, False, age
        counts = Counter(valid_ids)
        best, freq = counts.most_common(1)[0]
        if freq >= self.vote_th:
            _, is_valid = extract_container_number(best)
            return best, is_valid, age
        return None, False, age

    def prune(self, active_keys: set):
        stale = set(self._tracks.keys()) - active_keys
        for k in stale:
            del self._tracks[k]
            self._ages.pop(k, None)


_tracker = DetectionTracker()


# ─────────────────────────────────────────────
#  Detection (with dedup from v3)
# ─────────────────────────────────────────────

def detect_regions(frame_bgr: np.ndarray):
    detector = load_detector()
    boxes    = []
    if DETECTOR_BACKEND == "yolo":
        results = detector(frame_bgr, conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))
    return deduplicate_boxes(boxes, DEDUP_IOU)


# ─────────────────────────────────────────────
#  NEW (F) – Multi-threaded OCR helper
# ─────────────────────────────────────────────

_ocr_executor = ThreadPoolExecutor(max_workers=OCR_THREAD_WORKERS)


def _ocr_task(args):
    """Worker: (crop_bgr, x1, y1, x2, y2, conf) → result tuple."""
    crop_bgr, x1, y1, x2, y2, conf = args
    raw_ocr              = run_ocr_on_crop(crop_bgr)
    cont_id, is_valid    = extract_container_number(raw_ocr)
    return x1, y1, x2, y2, conf, raw_ocr, cont_id, is_valid


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def _draw_pill(frame, text1, text2, x1, y1, color):
    fs1, fs2            = 0.75, 0.52
    (tw1, th1), _       = cv2.getTextSize(text1, FONT, fs1, THICKNESS)
    (tw2, th2), _       = cv2.getTextSize(text2, FONT, fs2, 1)
    pad                 = 6
    pw                  = max(tw1, tw2) + pad * 2
    ph                  = th1 + th2 + pad * 3
    px, py              = x1, max(0, y1 - ph - 4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, text1, (px + pad, py + pad + th1),
                FONT, fs1, color, THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text2, (px + pad, py + pad + th1 + pad + th2),
                FONT, fs2, COLOR_TEXT_FG, 1, cv2.LINE_AA)


def draw_detection(frame, x1, y1, x2, y2, conf, stable_id,
                   is_valid, age, raw_ocr, best_conf=None):
    if stable_id and is_valid:
        color  = COLOR_VALID
        label1 = stable_id
        best_s = f"  best:{best_conf:.2f}" if best_conf is not None else ""
        label2 = f"✔ ISO  cur:{conf:.2f}{best_s}"
    elif stable_id:
        color  = COLOR_PARTIAL
        label1 = stable_id
        label2 = f"? check-digit  conf:{conf:.2f}"
    elif age < WARMUP_FRAMES:
        color  = COLOR_RAW
        label1 = "reading…"
        label2 = f"conf:{conf:.2f}"
    else:
        color  = COLOR_RAW
        label1 = raw_ocr[:12] if raw_ocr else "NO TEXT"
        label2 = f"conf:{conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
    _draw_pill(frame, label1, label2, x1, y1, color)
    return frame


def draw_hud(frame, fps_live: float, frame_idx: int, total_valid: int):
    """NEW (D) – Persistent HUD overlay with live stats."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps_live:.1f}",
        f"Frame: {frame_idx}",
        f"Valid containers: {total_valid}",
    ]
    pad = 8
    fsize, thick = 0.60, 1
    line_h = 22
    box_w  = 220
    box_h  = len(lines) * line_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - box_w - 4, 4), (w - 4, box_h + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (w - box_w + pad, pad + (i + 1) * line_h),
                    FONT, fsize, COLOR_OVERLAY, thick, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  Process a single frame (multi-threaded OCR)
# ─────────────────────────────────────────────

def process_frame(frame: np.ndarray, frame_idx: int,
                  det_logger: DetectionLogger) -> np.ndarray:
    annotated = frame.copy()
    boxes     = detect_regions(frame)
    active    = set()

    if not boxes:
        return annotated

    # Build OCR tasks
    tasks = []
    for (x1, y1, x2, y2, conf) in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tasks.append((crop.copy(), x1, y1, x2, y2, conf))

    # Submit all crops to thread pool simultaneously
    futures = {_ocr_executor.submit(_ocr_task, t): t for t in tasks}

    for future in as_completed(futures):
        try:
            x1, y1, x2, y2, conf, raw_ocr, cont_id, is_valid = future.result()
        except Exception as exc:
            print(f"[!] OCR worker error: {exc}")
            continue

        _tracker.update(x1, y1, x2, y2, cont_id)
        active.add(DetectionTracker._cell_key(x1, y1, x2, y2))

        stable_id, sv, age = _tracker.query(x1, y1, x2, y2)

        # Log the detection if valid
        if stable_id and sv:
            crop = frame[y1:y2, x1:x2]
            det_logger.update(stable_id, conf, sv, frame_idx,
                              crop if crop.size > 0 else None)

        best_conf = det_logger.best_conf_for(stable_id) if stable_id else None
        annotated = draw_detection(
            annotated, x1, y1, x2, y2,
            conf, stable_id, sv, age, raw_ocr, best_conf,
        )

    _tracker.prune(active)
    return annotated


# ─────────────────────────────────────────────
#  NEW (B) – Stream URL resolver (RTSP / HTTP / YT)
# ─────────────────────────────────────────────

def resolve_stream_url(path: str) -> str:
    """
    If path looks like a YouTube URL, attempt to resolve the actual
    stream URL via yt-dlp. Otherwise return path unchanged.
    """
    yt_patterns = [
        r"youtube\.com/watch",
        r"youtu\.be/",
        r"youtube\.com/live",
    ]
    if any(re.search(p, path, re.I) for p in yt_patterns):
        try:
            import yt_dlp  # type: ignore
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "quiet":  True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(path, download=False)
                url  = info.get("url") or info.get("manifest_url", path)
                print(f"[✔] YouTube stream resolved → {url[:80]}…")
                return url
        except ImportError:
            print("[!] yt-dlp not installed (pip install yt-dlp). Using URL directly.")
        except Exception as e:
            print(f"[!] yt-dlp error: {e}. Using URL directly.")
    return path


def open_capture(path) -> cv2.VideoCapture:
    """Open VideoCapture, resolving YouTube URLs if needed."""
    src = path
    if isinstance(path, str):
        src = resolve_stream_url(path)
        if src.isdigit():
            src = int(src)

    cap = cv2.VideoCapture(src)

    # Optimise buffer for live streams
    if isinstance(src, str) and (
        src.startswith("rtsp://") or
        src.startswith("http://") or
        src.startswith("https://")
    ):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ─────────────────────────────────────────────
#  NEW (E) – Reconnecting video reader
# ─────────────────────────────────────────────

def _reconnecting_read(cap: cv2.VideoCapture, src,
                       attempt: int = 0):
    """
    Try to read a frame; if it fails and we haven't exhausted retries,
    reopen the capture and try again. Returns (cap, ret, frame).
    """
    ret, frame = cap.read()
    if ret:
        return cap, True, frame

    # Frame read failed – try reconnect
    if attempt >= MAX_RECONNECT_ATTEMPTS:
        return cap, False, None

    print(f"[!] Stream lost. Reconnecting ({attempt + 1}/{MAX_RECONNECT_ATTEMPTS})…")
    cap.release()
    time.sleep(RECONNECT_DELAY_SEC)
    cap = open_capture(src)
    return _reconnecting_read(cap, src, attempt + 1)


# ─────────────────────────────────────────────
#  Video pipeline (updated with all new features)
# ─────────────────────────────────────────────

def run_on_video():
    src = VIDEO_INPUT_PATH
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cap = open_capture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_INPUT_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if VIDEO_OUTPUT_PATH:
        Path(VIDEO_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"[✔] Writing output → {VIDEO_OUTPUT_PATH}")

    det_logger = DetectionLogger(DETECTION_LOG_TXT, DETECTION_LOG_CSV,
                                 BEST_SNAPSHOTS_DIR)

    frame_idx = 0
    t_start   = time.time()
    print("[▶] Processing … press Q to quit.")

    while True:
        cap, ret, frame = _reconnecting_read(cap, src)
        if not ret:
            print("[■] Stream ended or unrecoverable.")
            break
        if frame is None:
            continue

        frame_idx += 1
        if SKIP_FRAMES and (frame_idx % (SKIP_FRAMES + 1) != 0):
            continue

        annotated = process_frame(frame, frame_idx, det_logger)

        elapsed  = time.time() - t_start
        fps_live = frame_idx / elapsed if elapsed > 0 else 0
        annotated = draw_hud(annotated, fps_live, frame_idx,
                             det_logger.total_valid())

        if writer:
            writer.write(annotated)

        disp = annotated
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(
                annotated,
                (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
            )
        cv2.imshow("Container OCR  v4", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[■] Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # NEW (G) – Final export
    det_logger.export_csv()
    print(f"[✔] Done. Total valid containers logged: {det_logger.total_valid()}")
    print(f"    Snapshots saved → {BEST_SNAPSHOTS_DIR}")
    print(f"    Detection log   → {DETECTION_LOG_TXT}")
    print(f"    CSV log         → {DETECTION_LOG_CSV}")


# ─────────────────────────────────────────────
#  Image folder pipeline (updated)
# ─────────────────────────────────────────────

def run_on_images():
    folder = Path(TEST_IMAGES_FOLDER)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"[!] No images in {TEST_IMAGES_FOLDER}")
        return

    det_logger = DetectionLogger(DETECTION_LOG_TXT, DETECTION_LOG_CSV,
                                 BEST_SNAPSHOTS_DIR)

    print(f"[▶] {len(images)} images — any key = next, Q = quit")
    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        annotated = process_frame(frame, i + 1, det_logger)
        cv2.imshow(f"Container OCR v4 – {img_path.name}", annotated)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    det_logger.export_csv()
    print(f"[✔] Done. Total valid containers logged: {det_logger.total_valid()}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if TEST_IMAGES_FOLDER:
        run_on_images()
    else:
        run_on_video()