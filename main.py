import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PLATE_MODEL_PATH = BASE_DIR / "license_plate_detector.pt"
VIDEO_DIRECTORIES = (
    BASE_DIR / "videos",
    BASE_DIR / "video",
    BASE_DIR / "video_1",
)
OUTPUT_DIR = BASE_DIR / "outputs"
PLATE_CROP_DIR = OUTPUT_DIR / "plate_crops"
OCR_MODEL_DIR = BASE_DIR / "ocr_models"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if GPU_AVAILABLE else "cpu"

# The stronger model is used on the RTX GPU. The CPU fallback keeps playback usable.
VEHICLE_MODEL_PATH = BASE_DIR / ("yolo11s.pt" if GPU_AVAILABLE else "yolo11n.pt")
VEHICLE_IMAGE_SIZE = 704 if GPU_AVAILABLE else 640
PLATE_IMAGE_SIZE = 640
VEHICLE_CONFIDENCE = 0.25
PLATE_CONFIDENCE = 0.20
PLATE_DETECTION_INTERVAL = 2
OCR_RETRY_INTERVAL = 10
MIN_OCR_CONFIDENCE = 0.35
PLAYBACK_SPEEDS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

# COCO class ids used by YOLO for road vehicles, including motorcycles.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CLASS_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 200, 0),
    "truck": (255, 0, 255),
}

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UA", "UP", "WB", "BH"
}

CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "U": "0", "C": "0",
    "I": "1", "T": "1", "L": "1",
    "J": "7", "Y": "7",
    "Z": "2",
    "E": "3",
    "A": "4", "H": "4",
    "S": "5",
    "G": "6", "B": "8",
    "P": "9",
}

# In series positions, 'I' and 'O' are not issued by RTO in India to prevent confusion with 1 and 0.
DIGIT_TO_SERIES_CHAR = {
    "0": "D", "O": "D", "Q": "D", "U": "D",
    "1": "L", "I": "L",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}

DIGIT_TO_CHAR = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"
}


class IdentityRegistry:
    def __init__(self):
        self.next_track_number = 1
        self.plate_to_reid = {}
        self.next_reid_number = 1

    def new_track_id(self):
        track_id = f"VEH-{self.next_track_number:05d}"
        self.next_track_number += 1
        return track_id

    def reid_for_plate(self, plate_text):
        if plate_text not in self.plate_to_reid:
            self.plate_to_reid[plate_text] = f"REID-{self.next_reid_number:05d}"
            self.next_reid_number += 1
        return self.plate_to_reid[plate_text]


def main():
    validate_project_files()
    configure_torch()
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLATE_CROP_DIR.mkdir(exist_ok=True)

    video_paths = find_video_paths()
    if not video_paths:
        folders = ", ".join(str(folder) for folder in VIDEO_DIRECTORIES)
        raise FileNotFoundError(f"No videos found in: {folders}")

    selected_video_paths = choose_videos(video_paths)
    if not selected_video_paths:
        return

    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))
    ocr_reader = create_ocr_reader()

    acceleration = "RTX GPU" if GPU_AVAILABLE else "CPU fallback"
    print(f"Running on: {acceleration}")
    print(f"Vehicle model: {VEHICLE_MODEL_PATH.name}")

    registry = IdentityRegistry()
    results_path = OUTPUT_DIR / "unique_vehicles.csv"
    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(
            results_file,
            fieldnames=[
                "vehicle_id",
                "re_id",
                "plate_text",
                "ocr_confidence",
                "video",
                "frame",
                "vehicle_type",
                "vehicle_confidence",
                "plate_detected",
            ],
        )
        writer.writeheader()

        video_index = 0
        while 0 <= video_index < len(selected_video_paths):
            action = play_video(
                vehicle_model,
                plate_model,
                ocr_reader,
                selected_video_paths[video_index],
                writer,
                registry,
                video_index,
                len(selected_video_paths),
            )
            if action == "quit":
                break
            if action == "previous":
                video_index = max(video_index - 1, 0)
            else:
                video_index += 1

    cv2.destroyAllWindows()
    print(f"Saved vehicle and plate results to: {results_path}")


def configure_torch():
    if GPU_AVAILABLE:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def validate_project_files():
    for required_path in (VEHICLE_MODEL_PATH, PLATE_MODEL_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file or folder is missing: {required_path}")


def find_video_paths():
    video_paths = []
    for directory in VIDEO_DIRECTORIES:
        if directory.is_dir():
            video_paths.extend(
                path for path in directory.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
            )
    return sorted(set(video_paths), key=lambda path: (path.parent.name.lower(), path.name.lower()))


def choose_videos(video_paths):
    window_name = "ANPR Video Selector"
    selected_index = 0
    marked_indexes = set()
    visible_rows = 14

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 940, 620)

    while True:
        panel = np.full((620, 940, 3), (28, 30, 35), dtype=np.uint8)
        cv2.putText(
            panel,
            f"Video library ({len(video_paths)} clips)",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (240, 240, 240),
            2,
        )
        cv2.putText(
            panel,
            f"Marked: {len(marked_indexes)}",
            (30, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (160, 200, 255),
            1,
        )

        first_index = max(0, min(selected_index - visible_rows // 2, len(video_paths) - visible_rows))
        for row, index in enumerate(range(first_index, min(first_index + visible_rows, len(video_paths)))):
            top = 108 + row * 34
            is_current = index == selected_index
            is_marked = index in marked_indexes
            if is_current:
                cv2.rectangle(panel, (20, top - 24), (920, top + 8), (55, 96, 130), -1)

            marker = "[x]" if is_marked else "[ ]"
            filename = video_paths[index].name
            cv2.putText(
                panel,
                f"{marker} {index + 1:03d}  {filename}",
                (38, top),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255) if is_current else (205, 205, 205),
                1,
            )

        cv2.imshow(window_name, panel)
        raw_key = cv2.waitKeyEx(30)
        key = raw_key & 0xFF
        if key in (13, 10):
            indexes = sorted(marked_indexes) if marked_indexes else [selected_index]
            cv2.destroyWindow(window_name)
            return [video_paths[index] for index in indexes]
        if key == ord("a"):
            cv2.destroyWindow(window_name)
            return video_paths
        if key == ord(" "):
            if selected_index in marked_indexes:
                marked_indexes.remove(selected_index)
            else:
                marked_indexes.add(selected_index)
        if raw_key in (2490368, 82):  # Up arrow on Windows/OpenCV
            selected_index = max(0, selected_index - 1)
        if raw_key in (2621440, 84):  # Down arrow on Windows/OpenCV
            selected_index = min(len(video_paths) - 1, selected_index + 1)
        if key in (27, ord("q")) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            return []


def create_ocr_reader():
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "EasyOCR is not installed. Install the project requirements and run main.py again."
        ) from error

    OCR_MODEL_DIR.mkdir(exist_ok=True)
    return easyocr.Reader(
        ["en"],
        gpu=GPU_AVAILABLE,
        model_storage_directory=str(OCR_MODEL_DIR),
        verbose=False,
    )


STATE_PREFIX_CORRECTIONS = {
    "OP": "UP", "VP": "UP", "0P": "UP", "IP": "UP", "JP": "UP", "LP": "UP", "CP": "UP", "UP": "UP",
    "0L": "DL", "OL": "DL", "D1": "DL", "DI": "DL", "DJ": "DL", "OI": "DL", "QL": "DL", "IL": "DL", "1L": "DL", "CL": "DL", "UL": "DL", "DL": "DL",
    "M1": "MH", "MR": "MH", "M4": "MH", "NH": "MH", "MH": "MH",
    "K4": "KA", "K8": "KA", "KA": "KA",
    "HB": "HR", "HA": "HR", "HR": "HR",
    "6J": "GJ", "G1": "GJ", "GI": "GJ", "GJ": "GJ",
    "8J": "RJ", "R1": "RJ", "RI": "RJ", "RJ": "RJ",
    "TM": "TN", "IN": "TN", "1N": "TN", "TN": "TN",
    "W8": "WB", "VB": "WB", "WP": "WB", "UB": "WB", "WB": "WB",
    "T5": "TS", "IS": "TS", "1S": "TS", "TS": "TS",
    "A8": "AP", "4P": "AP", "AP": "AP",
    "K1": "KL", "KI": "KL", "KL": "KL",
    "P8": "PB", "PB": "PB",
    "B8": "BR", "8R": "BR", "BR": "BR",
}


def normalize_state_code(code: str) -> str:
    code = code.upper()
    if code in INDIAN_STATE_CODES:
        return code
    return STATE_PREFIX_CORRECTIONS.get(code, code)


def correct_chars(text: str, target_types: str) -> str:
    """
    Given a text and target_types string where:
    'L' means general Letter, 'S' means Series Letter (O->D, 0->D, 1->L), 'D' means Digit.
    """
    if len(text) != len(target_types):
        return text
    result = []
    for ch, t in zip(text, target_types):
        if t == "D":
            result.append(CHAR_TO_DIGIT.get(ch, ch))
        elif t == "S":
            result.append(DIGIT_TO_SERIES_CHAR.get(ch, ch))
        elif t == "L":
            result.append(DIGIT_TO_CHAR.get(ch, ch))
        else:
            result.append(ch)
    return "".join(result)


def parse_and_score_plate(raw_text: str):
    """
    Evaluates raw OCR text against all standard Indian license plate syntactic models
    and returns (best_formatted_text, confidence_bonus, pattern_name).
    """
    if not raw_text:
        return "", 0.0, "empty"

    text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    # Strip prefixes like IND, INDIA, HSRP, HRSP
    for pfx in ["INDIA", "HSRP", "HRSP", "IND"]:
        if text.startswith(pfx) and len(text) >= len(pfx) + 6:
            text = text[len(pfx):]
            break

    # Strip leading stray noise chars like 'F' in 'FOL4CAS7269' or 'E' in 'EOL4...'
    if len(text) >= 11 and text[0] in ("F", "E", "I", "T", "C", "P") and text[1:3] in ("DL", "0L", "OL", "UP", "MH", "HR", "WB", "KA"):
        text = text[1:]

    if len(text) > 12:
        text = text[:12]

    if len(text) < 4:
        return "", 0.0, "too_short"

    candidates = []

    # 1. Delhi 10-char: DL + 1-digit RTO + 1-letter Category + 2-letter Series + 4 digits (e.g. DL 4C AS 7269, DL 1L AG 4975)
    # Structure: LL D S SS DDDD (Total 10 chars)
    if len(text) == 10:
        st = normalize_state_code(text[:2])
        if st == "DL":
            c_dl = correct_chars(text, "LLDSSSDDDD")
            cand = f"DL{c_dl[2]}{c_dl[3]}{c_dl[4:6]}{c_dl[6:]}"
            rto_digit = cand[2]
            cat_char = cand[3]
            series_chars = cand[4:6]
            number_digits = cand[6:]
            if rto_digit.isdigit() and cat_char.isalpha() and series_chars.isalpha() and number_digits.isdigit():
                candidates.append((cand, 0.55, "Delhi_10_1RTO"))

    # 2. Standard 10-char: LL DD SS DDDD (e.g. UP16BZ4237, UP16DP7811, UP14MT6919, MH12DE1433)
    if len(text) == 10:
        c_std = correct_chars(text, "LLDDSSDDDD")
        state = normalize_state_code(c_std[:2])
        rto = c_std[2:4]
        series = c_std[4:6]
        digits = c_std[6:]
        cand = f"{state}{rto}{series}{digits}"
        if state.isalpha() and rto.isdigit() and series.isalpha() and digits.isdigit():
            bonus = 0.40 if state in INDIAN_STATE_CODES else 0.20
            candidates.append((cand, bonus, "Standard_10"))

    # 3. Bharat Series 10-char: DD BH DDDD SS (e.g. 22BH1234AA)
    if len(text) == 10 and ("BH" in text[1:4]):
        c_bh = correct_chars(text, "DDLLDDDDSS")
        if c_bh[:2].isdigit() and c_bh[2:4] == "BH" and c_bh[4:8].isdigit() and c_bh[8:].isalpha():
            candidates.append((c_bh, 0.45, "Bharat_10"))

    # 4. Standard 9-char formats:
    if len(text) == 9:
        st = normalize_state_code(text[:2])
        if st == "DL":
            # Delhi 9-char: LL D S S DDDD (e.g. DL 4 C A 1234)
            c_dl9 = correct_chars(text, "LLDSSDDDD")
            cand = f"DL{c_dl9[2]}{c_dl9[3]}{c_dl9[4]}{c_dl9[5:]}"
            if cand[2].isdigit() and cand[3:5].isalpha() and cand[5:].isdigit():
                candidates.append((cand, 0.45, "Delhi_9"))

        # Case B: Standard 9-char: LL DD S DDDD (e.g. MH 12 A 1234)
        c1 = correct_chars(text, "LLDDSDDDD")
        state1 = normalize_state_code(c1[:2])
        cand1 = f"{state1}{c1[2:]}"
        if state1.isalpha() and cand1[2:4].isdigit() and cand1[4].isalpha() and cand1[5:].isdigit():
            bonus = 0.35 if state1 in INDIAN_STATE_CODES else 0.15
            candidates.append((cand1, bonus, "Std_9_1Series"))

        # Case C: 2 series letters + 3 number digits: LL DD SS DDD (e.g. UP 80 ES 057)
        c3 = correct_chars(text, "LLDDSSDDD")
        state3 = normalize_state_code(c3[:2])
        cand3 = f"{state3}{c3[2:]}"
        if state3.isalpha() and cand3[2:4].isdigit() and cand3[4:6].isalpha() and cand3[6:].isdigit():
            bonus = 0.35 if state3 in INDIAN_STATE_CODES else 0.15
            candidates.append((cand3, bonus, "Std_9_3Num"))

    # 5. Standard 8-char: LL DD DDDD or LL D S DDDD (e.g. DL4C1234)
    if len(text) == 8:
        c = correct_chars(text, "LLDDDDDD")
        state = normalize_state_code(c[:2])
        cand = f"{state}{c[2:]}"
        if state.isalpha() and cand[2:].isdigit():
            bonus = 0.25 if state in INDIAN_STATE_CODES else 0.15
            candidates.append((cand, bonus, "Std_8"))

    # 6. Standard 11-char: LL DD SSS DDDD (e.g. DL 01 ABC 1234)
    if len(text) == 11:
        c = correct_chars(text, "LLDDSSSDDDD")
        state = normalize_state_code(c[:2])
        cand = f"{state}{c[2:]}"
        if state.isalpha() and cand[2:4].isdigit() and cand[4:7].isalpha() and cand[7:].isdigit():
            bonus = 0.35 if state in INDIAN_STATE_CODES else 0.20
            candidates.append((cand, bonus, "Std_11"))

    if not candidates:
        if len(text) >= 6:
            first_two = normalize_state_code(correct_chars(text[:2], "LL"))
            last_four = correct_chars(text[-4:], "DDDD")
            middle = text[2:-4]
            cand = first_two + middle + last_four
            bonus = 0.10 if first_two in INDIAN_STATE_CODES else 0.0
            return cand, bonus, "Fallback"
        return text, 0.0, "Raw"

    return max(candidates, key=lambda c: c[1])


def clean_and_format_plate(raw_text: str):
    cand, bonus, _ = parse_and_score_plate(raw_text)
    return cand, bonus


def assemble_ocr_detections(detections):
    """
    Groups OCR bounding boxes into lines (top-to-bottom) and columns (left-to-right),
    and produces joined candidate texts with aggregated confidence scores.
    """
    if not detections:
        return []

    parsed = []
    for item in detections:
        if len(item) == 3:
            box, text, conf = item
        elif len(item) == 2:
            box, text = item
            conf = 0.5
        else:
            continue

        clean = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not clean:
            continue

        box = np.array(box)
        min_x = float(np.min(box[:, 0]))
        max_x = float(np.max(box[:, 0]))
        min_y = float(np.min(box[:, 1]))
        max_y = float(np.max(box[:, 1]))
        center_y = (min_y + max_y) / 2.0
        center_x = (min_x + max_x) / 2.0
        box_h = max(max_y - min_y, 1.0)
        parsed.append({
            "text": clean,
            "conf": float(conf),
            "min_x": min_x,
            "min_y": min_y,
            "center_x": center_x,
            "center_y": center_y,
            "box_h": box_h,
        })

    if not parsed:
        return []

    # Sort parsed boxes by vertical center
    parsed.sort(key=lambda d: d["center_y"])
    avg_h = sum(d["box_h"] for d in parsed) / len(parsed)

    # Group into lines
    lines = []
    current_line = [parsed[0]]
    for item in parsed[1:]:
        if abs(item["center_y"] - current_line[0]["center_y"]) < (avg_h * 0.45):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    if current_line:
        lines.append(current_line)

    # Within each line, sort left-to-right
    full_text_parts = []
    conf_scores = []
    for line in lines:
        line.sort(key=lambda d: d["min_x"])
        line_text = "".join(d["text"] for d in line)
        full_text_parts.append(line_text)
        conf_scores.extend(d["conf"] for d in line)

    assembled_text = "".join(full_text_parts)
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    candidates = [(assembled_text, avg_conf)]
    for item in parsed:
        if len(item["text"]) >= 6:
            candidates.append((item["text"], item["conf"]))

    return candidates


def enhance_plate_variants(plate_crop):
    """
    Generates high-fidelity image enhancement variants specifically tuned for
    license plate character clarity and contrast.
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    h, w = plate_crop.shape[:2]
    target_h = max(100, min(160, int(h * 3.0)))
    scale = target_h / max(h, 1)
    target_w = max(220, int(w * scale))

    resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Variant 1: Bilateral Denoising + CLAHE + Unsharp Masking
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(denoised)
    sharp = cv2.addWeighted(clahe, 1.5, cv2.GaussianBlur(clahe, (0, 0), 1.5), -0.5, 0)

    # Variant 2: Adaptive Gaussian Thresholding
    adapt_thresh = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)

    # Variant 3: Otsu Binarization
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Variant 4: Inverted Otsu (for white-on-dark or yellow-black plates)
    otsu_inv = cv2.bitwise_not(otsu)

    # Variant 5: Normalized Contrast Grayscale
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return [
        resized,
        sharp,
        adapt_thresh,
        otsu,
        otsu_inv,
        norm,
    ]


def read_plate_text(ocr_reader, frame, plate_box):
    x1, y1, x2, y2, _ = plate_box
    h_frame, w_frame = frame.shape[:2]

    # Add 12% context padding around plate box to avoid clipping outer characters
    pad_x = int((x2 - x1) * 0.12)
    pad_y = int((y2 - y1) * 0.15)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w_frame, x2 + pad_x)
    cy2 = min(h_frame, y2 + pad_y)

    plate_crop = frame[cy1:cy2, cx1:cx2]
    if plate_crop.size == 0:
        return "", 0.0, 0.0, plate_crop

    variants = enhance_plate_variants(plate_crop)
    all_candidates = []

    for variant in variants:
        detections = ocr_reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            beamWidth=5,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            batch_size=1,
        )
        assembled = assemble_ocr_detections(detections)
        for raw_text, base_conf in assembled:
            formatted_text, format_bonus, _ = parse_and_score_plate(raw_text)
            if 6 <= len(formatted_text) <= 12:
                composite_score = base_conf + format_bonus
                all_candidates.append({
                    "text": formatted_text,
                    "conf": base_conf,
                    "score": composite_score,
                })

    if not all_candidates:
        return "", 0.0, 0.0, plate_crop

    # Consensus voting across top candidates if multiple valid 10-char candidates exist
    ten_char_cands = [c for c in all_candidates if len(c["text"]) == 10 and c["score"] >= 0.70]
    if len(ten_char_cands) >= 2:
        voted_chars = []
        for pos in range(10):
            char_votes = {}
            for c in ten_char_cands:
                ch = c["text"][pos]
                char_votes[ch] = char_votes.get(ch, 0.0) + c["score"]
            best_ch = max(char_votes.items(), key=lambda item: item[1])[0]
            voted_chars.append(best_ch)
        consensus_text = "".join(voted_chars)
        formatted_consensus, bonus_consensus, _ = parse_and_score_plate(consensus_text)
        avg_conf = sum(c["conf"] for c in ten_char_cands) / len(ten_char_cands)
        return formatted_consensus, avg_conf, avg_conf + bonus_consensus, plate_crop

    best_candidate = max(all_candidates, key=lambda c: c["score"])
    return best_candidate["text"], best_candidate["conf"], best_candidate["score"], plate_crop


def find_plate_for_vehicle(vehicle_box, plate_boxes):
    vx1, vy1, vx2, vy2 = vehicle_box
    vh = vy2 - vy1
    vw = vx2 - vx1
    # Allow expansion for bumper plates and YOLO jitter
    evx1 = vx1 - int(vw * 0.10)
    evy1 = vy1 - int(vh * 0.08)
    evx2 = vx2 + int(vw * 0.10)
    evy2 = vy2 + int(vh * 0.15)

    matching_plate = None
    best_confidence = -1.0

    for px1, py1, px2, py2, confidence in plate_boxes:
        plate_center_x = (px1 + px2) // 2
        plate_center_y = (py1 + py2) // 2
        is_inside_vehicle = evx1 <= plate_center_x <= evx2 and evy1 <= plate_center_y <= evy2

        if is_inside_vehicle and confidence > best_confidence:
            matching_plate = (px1, py1, px2, py2, confidence)
            best_confidence = confidence

    return matching_plate


def get_plate_boxes(plate_results):
    boxes = []
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
    return boxes


def reset_tracker(vehicle_model):
    predictor = getattr(vehicle_model, "predictor", None)
    if predictor is None or not hasattr(predictor, "trackers"):
        return

    for tracker in predictor.trackers:
        tracker.reset()


def play_video(
    vehicle_model,
    plate_model,
    ocr_reader,
    video_path,
    writer,
    registry,
    video_index,
    video_count,
):
    reset_tracker(vehicle_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping video because it could not be opened: {video_path.name}")
        return "next"

    print(f"Playing: {video_path.name}")
    frame_number = 0
    tracker_to_vehicle_id = {}
    vehicle_records = {}
    vehicle_best_plates = {}
    next_ocr_frame = {}
    cached_plate_boxes = []
    displayed_fps = 0.0
    playback_action = "next"
    paused = False
    playback_speed_index = PLAYBACK_SPEEDS.index(1.00)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                paused = False
            elif key == ord("n"):
                playback_action = "next"
                break
            elif key == ord("p"):
                playback_action = "previous"
                break
            elif key == ord("q"):
                playback_action = "quit"
                break
            elif key in (ord("-"), ord("_")):
                playback_speed_index = max(0, playback_speed_index - 1)
            elif key in (ord("+"), ord("=")):
                playback_speed_index = min(len(PLAYBACK_SPEEDS) - 1, playback_speed_index + 1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        started_at = time.perf_counter()
        frame_number += 1
        vehicle_results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=VEHICLE_CONFIDENCE,
            classes=list(VEHICLE_CLASSES),
            imgsz=VEHICLE_IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        plates_are_fresh = frame_number % PLATE_DETECTION_INTERVAL == 0
        if plates_are_fresh:
            plate_results = plate_model(
                frame,
                conf=PLATE_CONFIDENCE,
                imgsz=PLATE_IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )
            cached_plate_boxes = get_plate_boxes(plate_results)

        active_vehicle_count = 0
        for result in vehicle_results:
            for box in result.boxes:
                if box.id is None:
                    continue

                tracker_id = int(box.id[0])
                if tracker_id not in tracker_to_vehicle_id:
                    tracker_to_vehicle_id[tracker_id] = registry.new_track_id()

                vehicle_id = tracker_to_vehicle_id[tracker_id]
                class_id = int(box.cls[0])
                vehicle_type = VEHICLE_CLASSES[class_id]
                vehicle_confidence = float(box.conf[0])
                vehicle_box = tuple(map(int, box.xyxy[0]))
                matching_plate = find_plate_for_vehicle(vehicle_box, cached_plate_boxes)

                # Dual-scale detection: if plate not found on full-frame, detect inside vehicle crop
                if matching_plate is None and plates_are_fresh:
                    vx1, vy1, vx2, vy2 = vehicle_box
                    vw, vh = vx2 - vx1, vy2 - vy1
                    if vw >= 50 and vh >= 50:
                        v_crop = frame[max(0, vy1):min(frame.shape[0], vy2), max(0, vx1):min(frame.shape[1], vx2)]
                        if v_crop.size:
                            v_results = plate_model(v_crop, conf=0.15, imgsz=640, device=DEVICE, verbose=False)
                            v_boxes = get_plate_boxes(v_results)
                            if v_boxes:
                                px1, py1, px2, py2, pconf = v_boxes[0]
                                matching_plate = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2, pconf)

                active_vehicle_count += 1

                record = vehicle_records.setdefault(
                    vehicle_id,
                    {
                        "vehicle_id": vehicle_id,
                        "re_id": "",
                        "plate_text": "",
                        "ocr_confidence": "",
                        "video": video_path.name,
                        "frame": frame_number,
                        "vehicle_type": vehicle_type,
                        "vehicle_confidence": f"{vehicle_confidence:.2f}",
                        "plate_detected": "no",
                    },
                )

                if matching_plate is not None:
                    record["plate_detected"] = "yes"
                    plate_area = (matching_plate[2] - matching_plate[0]) * (matching_plate[3] - matching_plate[1])
                    best_info = vehicle_best_plates.get(vehicle_id, {"score": 0.0, "area": 0})

                    should_run_ocr = (
                        plates_are_fresh and (
                            best_info["score"] < 0.85 and frame_number >= next_ocr_frame.get(vehicle_id, 0)
                            or plate_area >= best_info["area"] * 1.20
                        )
                    )

                    if should_run_ocr:
                        plate_text, ocr_confidence, composite_score, plate_crop = read_plate_text(
                            ocr_reader, frame, matching_plate
                        )
                        next_ocr_frame[vehicle_id] = frame_number + OCR_RETRY_INTERVAL

                        if plate_text and ocr_confidence >= MIN_OCR_CONFIDENCE:
                            if composite_score > best_info["score"]:
                                record["plate_text"] = plate_text
                                record["ocr_confidence"] = f"{ocr_confidence:.2f}"
                                record["re_id"] = registry.reid_for_plate(plate_text)
                                record["frame"] = frame_number
                                save_plate_crop(plate_crop, record["re_id"])
                                vehicle_best_plates[vehicle_id] = {
                                    "score": composite_score,
                                    "area": plate_area,
                                    "plate_text": plate_text,
                                }

                draw_vehicle(frame, vehicle_box, record, vehicle_type, vehicle_confidence)
                if matching_plate is not None:
                    draw_plate(frame, matching_plate, record["plate_text"])

        frame_seconds = time.perf_counter() - started_at
        current_fps = 1 / frame_seconds if frame_seconds else 0.0
        displayed_fps = current_fps if displayed_fps == 0 else 0.9 * displayed_fps + 0.1 * current_fps
        draw_status(
            frame,
            f"{video_index + 1}/{video_count} | {video_path.name}",
            active_vehicle_count,
            len(tracker_to_vehicle_id),
            displayed_fps,
            PLAYBACK_SPEEDS[playback_speed_index],
        )
        cv2.imshow("Vehicle, Plate OCR, and Re-ID", frame)

        target_frame_seconds = 1 / (source_fps * PLAYBACK_SPEEDS[playback_speed_index])
        wait_milliseconds = max(1, int((target_frame_seconds - frame_seconds) * 1000))
        key = cv2.waitKey(wait_milliseconds) & 0xFF
        if key == ord("q"):
            playback_action = "quit"
            break
        if key == ord("n"):
            playback_action = "next"
            break
        if key == ord("p"):
            playback_action = "previous"
            break
        if key == ord(" "):
            paused = True
        if key in (ord("-"), ord("_")):
            playback_speed_index = max(0, playback_speed_index - 1)
        if key in (ord("+"), ord("=")):
            playback_speed_index = min(len(PLAYBACK_SPEEDS) - 1, playback_speed_index + 1)
        if cv2.getWindowProperty("Vehicle, Plate OCR, and Re-ID", cv2.WND_PROP_VISIBLE) < 1:
            playback_action = "quit"
            break

    cap.release()
    writer.writerows(vehicle_records.values())
    return playback_action


def save_plate_crop(plate_crop, re_id):
    if plate_crop.size:
        cv2.imwrite(str(PLATE_CROP_DIR / f"{re_id}.jpg"), plate_crop)


def draw_vehicle(frame, vehicle_box, record, vehicle_type, confidence):
    x1, y1, x2, y2 = vehicle_box
    color = CLASS_COLORS[vehicle_type]
    identity = record["re_id"] or record["vehicle_id"]
    label = f"{identity} | {vehicle_type} {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_plate(frame, plate_box, plate_text):
    x1, y1, x2, y2, confidence = plate_box
    label = plate_text or f"plate {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, min(y2 + 20, frame.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )


def draw_status(frame, video_name, active_count, unique_count, fps, playback_speed):
    device_label = "GPU" if GPU_AVAILABLE else "CPU"
    cv2.putText(
        frame,
        f"{device_label} | {fps:.1f} FPS | Speed: {playback_speed:.2f}x | Active: {active_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        video_name,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


if __name__ == "__main__":
    main()

