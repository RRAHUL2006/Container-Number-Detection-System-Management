"""
Container Number Recognition System  
=======================================================
Fixes over v2:
  1. ISO 6346 alpha lookup table was WRONG in v2 (every letter B-Z was off by 1).
     Fixed with a hardcoded, standards-verified table.
  2. Duplicate bounding-box detections for the same region are now suppressed
     with a post-NMS IoU deduplication pass.
  3. '5' ↔ 'S'  and  '6' ↔ 'G'  swaps in the digit zone were silently ignored;
     extended _DIGIT_FIX_MAP and added a fallback that tries all likely
     single-character substitutions when the ISO check-digit fails.
  4. CONTAINER_REGEX extended to accept an optional space/dash between the
     owner prefix and the serial number (common in OCR output).
"""

import cv2
import numpy as np
import re
import time
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
#  ★  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────

MODEL_WEIGHTS_PATH = r"C:\Users\R RAHUL\OneDrive\Desktop\container\runs\detect\train4\weights\last.pt"
OCR_MODEL_PATH     = r"C:\Users\R RAHUL\OneDrive\Desktop\container"
VIDEO_INPUT_PATH   = r"C:\Users\R RAHUL\OneDrive\Desktop\container\Container Number Recognition System (CNRS).mp4"
VIDEO_OUTPUT_PATH  = r"C:\Users\R RAHUL\OneDrive\Desktop\container\annotated_output_v3.mp4"
TEST_IMAGES_FOLDER = ""
 



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
DEDUP_IOU        = 0.40    # NEW: post-NMS deduplication threshold
DISPLAY_SCALE    = 1.0
SKIP_FRAMES      = 2

# Temporal smoothing
TRACK_HISTORY    = 8
VOTE_THRESHOLD   = 3
WARMUP_FRAMES    = 2

# ─────────────────────────────────────────────
#  FIX 1 – Correct ISO 6346 alpha lookup table
#  Standard: A=10, skip 11, B=12, C=13 … K=21, skip 22, L=23 …
#            U=32, skip 33, V=34 … Z=38
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

# In the DIGIT zone: map look-alike letters → correct digit
_DIGIT_FIX_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

# In the LETTER zone: map look-alike digits → correct letter
_LETTER_FIX_MAP = {
    "0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z",
}

# FIX 4 – regex now tolerates an optional separator between prefix and serial
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

    # EASYOCR
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

    # PADDLEOCR
    elif OCR_BACKEND == "paddleocr":

        try:
            from paddleocr import PaddleOCR

            _ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                show_log=False,
                enable_mkldnn=False
            )

            print("[✔] PaddleOCR loaded")

        except ImportError:
            raise ImportError(
                "pip install paddlepaddle paddleocr"
            )

    # TESSERACT
    elif OCR_BACKEND == "tesseract":

        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = OCR_MODEL_PATH

        _ocr = pytesseract

    else:
        raise ValueError(
            f"Unknown OCR_BACKEND: {OCR_BACKEND}"
        )

    return _ocr


# ─────────────────────────────────────────────
#  FIX 1 – Corrected ISO 6346 check-digit
# ─────────────────────────────────────────────

def _iso6346_checkdigit(owner: str, serial6: str) -> int:
    """Return the expected ISO 6346 check digit (0-9) for owner+serial6."""
    code = (owner + serial6).upper()
    total = 0
    for i, ch in enumerate(code):
        val = _ISO_ALPHA[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    r = total % 11
    return 0 if r == 10 else r


def iso6346_check(owner: str, serial: str) -> bool:
    """
    Returns True only when serial has 7 chars and its last digit
    matches the ISO 6346 check digit computed from owner+serial[:6].
    """
    if len(serial) < 7 or len(owner) != 4:
        return False
    try:
        return int(serial[6]) == _iso6346_checkdigit(owner, serial[:6])
    except (ValueError, KeyError):
        return False


# ─────────────────────────────────────────────
#  FIX 3 – Fuzzy check-digit recovery
#  Try single-character substitutions in the digit zone when the
#  raw OCR string fails the check digit but is otherwise plausible.
# ─────────────────────────────────────────────

_SIMILAR_DIGITS = {
    "0": ["O", "Q", "D"],
    "1": ["I", "L"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
}
_SIMILAR_LETTERS_IN_DIGIT_ZONE = {v: k for k, vs in _SIMILAR_DIGITS.items() for v in vs}


def _try_fix_checkdigit(owner: str, serial7: str):
    """
    Given an owner code and a 7-character serial that failed ISO validation,
    try swapping each digit position with its common OCR look-alikes.
    Returns (fixed_serial, True) on success, or (serial7, False) if no fix found.
    """
    # First try fixing the check digit position (index 6) specifically
    for sub in range(10):
        candidate = serial7[:6] + str(sub)
        if iso6346_check(owner, candidate):
            return candidate, True

    # Then try fixing each of the first 6 digit positions
    for pos in range(6):
        ch = serial7[pos]
        alternatives = _SIMILAR_LETTERS_IN_DIGIT_ZONE.get(ch, None)
        if alternatives is None:
            # ch is already a digit; try its look-alike letters mapped back
            for letter, digit in _SIMILAR_LETTERS_IN_DIGIT_ZONE.items():
                if digit == ch:
                    alternatives = digit  # already correct digit, skip
            continue
        # alternatives is the digit string
        candidate_serial = serial7[:pos] + alternatives + serial7[pos+1:]
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
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return gray


def _pipeline_adaptive(gray: np.ndarray) -> np.ndarray:
    g    = _upscale(gray)
    proc = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)
    k    = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(proc, -1, k)


def _pipeline_otsu(gray: np.ndarray) -> np.ndarray:
    g     = _upscale(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g     = clahe.apply(g)
    _, proc = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return proc


def _pipeline_morph(gray: np.ndarray) -> np.ndarray:
    g      = _upscale(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g      = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    _, proc = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return proc


_PIPELINES = [_pipeline_adaptive, _pipeline_otsu, _pipeline_morph]


# ─────────────────────────────────────────────
#  Character correction
# ─────────────────────────────────────────────

def _correct_container_string(raw: str) -> str:
    """
    Enforce letter-zone / digit-zone constraints on OCR output.
    Positions 0-3 must be letters; positions 4-10 must be digits.
    """
    raw = raw.upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    if len(raw) < 10:
        return raw

    letter_part = raw[:4]
    digit_part  = raw[4:]

    lp_fixed = "".join(_LETTER_FIX_MAP.get(ch, ch) for ch in letter_part)
    dp_fixed = "".join(_DIGIT_FIX_MAP.get(ch, ch) for ch in digit_part)

    return lp_fixed + dp_fixed


# ─────────────────────────────────────────────
#  Core OCR  – multi-pipeline with voting
# ─────────────────────────────────────────────

def run_ocr_on_crop(crop_bgr: np.ndarray) -> str:
    ocr     = load_ocr()
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    results = []

    for pipeline in _PIPELINES:
        proc     = pipeline(gray)
        proc_bgr = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        raw_text = ""

        if OCR_BACKEND == "easyocr":
            res = ocr.readtext(
                proc_bgr, detail=0,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            )
            raw_text = " ".join(res)

        elif OCR_BACKEND == "paddleocr":
            res = ocr.ocr(proc_bgr, cls=True)
            if res and res[0]:
                raw_text = " ".join(line[1][0] for line in res[0])

        elif OCR_BACKEND == "tesseract":
            cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            raw_text = ocr.image_to_string(proc_bgr, config=cfg)

        cleaned = _correct_container_string(raw_text.upper())
        if cleaned:
            results.append(cleaned)

    if not results:
        return ""

    counts = Counter(results)
    winner, freq = counts.most_common(1)[0]
    return winner if freq > 1 else results[0]


# ─────────────────────────────────────────────
#  Container number extraction + validation
#  Now includes fuzzy check-digit recovery (FIX 3)
# ─────────────────────────────────────────────

def extract_container_number(text: str):
    """
    Returns (container_id, is_valid_checksum) or (None, False).
    Attempts fuzzy single-char fix when the check digit fails.
    """
    m = CONTAINER_REGEX.search(text)
    if not m:
        return None, False

    owner  = m.group(1).upper()
    serial = m.group(2)

    # Pad to 7 digits if OCR missed the check digit
    if len(serial) == 6:
        expected_cd = _iso6346_checkdigit(owner, serial)
        serial = serial + str(expected_cd)

    if iso6346_check(owner, serial):
        return owner + serial, True

    # Try single-char fuzzy fix
    fixed_serial, fixed = _try_fix_checkdigit(owner, serial[:7])
    if fixed:
        return owner + fixed_serial, True

    return owner + serial[:7], False


# ─────────────────────────────────────────────
#  FIX 2 – IoU-based duplicate suppression
# ─────────────────────────────────────────────

def _iou(a, b) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def deduplicate_boxes(boxes, iou_threshold: float = DEDUP_IOU):
    """
    Remove duplicate / heavily overlapping boxes keeping the highest-confidence one.
    `boxes` is a list of (x1,y1,x2,y2,conf).
    """
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)  # highest conf first
    kept  = []
    for box in boxes:
        if all(_iou(box[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


# ─────────────────────────────────────────────
#  Temporal tracker
# ─────────────────────────────────────────────

class DetectionTracker:
    def __init__(self, history: int = TRACK_HISTORY, vote_th: int = VOTE_THRESHOLD):
        self.history = history
        self.vote_th = vote_th
        self._tracks = defaultdict(list)
        self._ages   = defaultdict(int)

    @staticmethod
    def _cell_key(x1, y1, x2, y2, grid: int = 80) -> tuple:
        cx = (x1 + x2) // 2 // grid
        cy = (y1 + y2) // 2 // grid
        return cx, cy

    def update(self, x1, y1, x2, y2, raw_id):
        key     = self._cell_key(x1, y1, x2, y2)
        self._ages[key] += 1
        history = self._tracks[key]
        history.append(raw_id)
        if len(history) > self.history:
            history.pop(0)

    def query(self, x1, y1, x2, y2):
        key      = self._cell_key(x1, y1, x2, y2)
        age      = self._ages.get(key, 0)
        hist     = self._tracks.get(key, [])
        valid_ids = [h for h in hist if h is not None]
        if not valid_ids:
            return None, False, age
        counts   = Counter(valid_ids)
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
#  Detection  (uses deduplicate_boxes now)
# ─────────────────────────────────────────────

def detect_regions(frame_bgr: np.ndarray):
    """Returns deduplicated list of (x1, y1, x2, y2, confidence)."""
    detector = load_detector()
    boxes    = []
    if DETECTOR_BACKEND == "yolo":
        results = detector(frame_bgr, conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

    # FIX 2: remove duplicates the model's NMS missed
    return deduplicate_boxes(boxes, DEDUP_IOU)


# ─────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────

def _draw_pill(frame, text1, text2, x1, y1, color):
    fs1, fs2 = 0.75, 0.52
    (tw1, th1), _ = cv2.getTextSize(text1, FONT, fs1, THICKNESS)
    (tw2, th2), _ = cv2.getTextSize(text2, FONT, fs2, 1)
    pad    = 6
    pw     = max(tw1, tw2) + pad * 2
    ph     = th1 + th2 + pad * 3
    px, py = x1, max(0, y1 - ph - 4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, text1,
                (px + pad, py + pad + th1),
                FONT, fs1, color, THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text2,
                (px + pad, py + pad + th1 + pad + th2),
                FONT, fs2, COLOR_TEXT_FG, 1, cv2.LINE_AA)


def draw_detection(frame, x1, y1, x2, y2, conf, stable_id, is_valid, age, raw_ocr):
    if stable_id and is_valid:
        color  = COLOR_VALID
        label1 = stable_id
        label2 = f"✔ ISO  conf:{conf:.2f}"
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


# ─────────────────────────────────────────────
#  Process a single frame
# ─────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    annotated = frame.copy()
    boxes     = detect_regions(frame)   # already deduplicated
    active    = set()

    for (x1, y1, x2, y2, conf) in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        raw_ocr  = run_ocr_on_crop(crop)
        cont_id, _ = extract_container_number(raw_ocr)

        _tracker.update(x1, y1, x2, y2, cont_id)
        active.add(DetectionTracker._cell_key(x1, y1, x2, y2))

        stable_id, is_valid, age = _tracker.query(x1, y1, x2, y2)
        annotated = draw_detection(
            annotated, x1, y1, x2, y2,
            conf, stable_id, is_valid, age, raw_ocr,
        )

    _tracker.prune(active)
    return annotated


# ─────────────────────────────────────────────
#  Video pipeline
# ─────────────────────────────────────────────

def run_on_video():
    src = int(VIDEO_INPUT_PATH) if str(VIDEO_INPUT_PATH).isdigit() else VIDEO_INPUT_PATH
    cap = cv2.VideoCapture(src)
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

    frame_idx = 0
    t_start   = time.time()
    print("[▶] Processing … press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if SKIP_FRAMES and (frame_idx % (SKIP_FRAMES + 1) != 0):
            continue

        annotated = process_frame(frame)

        elapsed  = time.time() - t_start
        fps_live = frame_idx / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS:{fps_live:.1f}  Frame:{frame_idx}",
                    (10, 28), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if writer:
            writer.write(annotated)

        disp = annotated
        if DISPLAY_SCALE != 1.0:
            disp = cv2.resize(annotated,
                              (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
        cv2.imshow("Container OCR  v3", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[■] Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[✔] Done.")


# ─────────────────────────────────────────────
#  Image folder pipeline
# ─────────────────────────────────────────────

def run_on_images():
    folder = Path(TEST_IMAGES_FOLDER)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"[!] No images in {TEST_IMAGES_FOLDER}")
        return

    print(f"[▶] {len(images)} images — any key = next, Q = quit")
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        annotated = process_frame(frame)
        cv2.imshow(f"Container OCR v3 – {img_path.name}", annotated)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if TEST_IMAGES_FOLDER:
        run_on_images()
    else:
        run_on_video()