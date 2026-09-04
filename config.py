from pathlib import Path
import torch

# Base project paths
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
LPRNET_MODEL_PATH = OCR_MODEL_DIR / "best_lprnet.pth"
CRNN_MODEL_PATH = BASE_DIR / "best_crnn_bilstm.pth"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Hardware acceleration settings
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if GPU_AVAILABLE else "cpu"

INDIAN_VEHICLE_MODEL = BASE_DIR / "indian_vehicles_yolo11.pt"
INDIAN_PLATE_MODEL = BASE_DIR / "indian_plate_yolo11.pt"

VEHICLE_MODEL_PATH = BASE_DIR / "yolo11s.pt" if (BASE_DIR / "yolo11s.pt").exists() else (BASE_DIR / "yolo11n.pt")
PLATE_MODEL_PATH = INDIAN_PLATE_MODEL if INDIAN_PLATE_MODEL.exists() else (BASE_DIR / "license_plate_detector.pt")
VEHICLE_IMAGE_SIZE = 704 if GPU_AVAILABLE else 640
PLATE_IMAGE_SIZE = 640

# Confidence & Interval Thresholds
VEHICLE_CONFIDENCE = 0.20
PLATE_CONFIDENCE = 0.25
PLATE_DETECTION_INTERVAL = 2
OCR_RETRY_INTERVAL = 8
OCR_FAST_INTERVAL = 3
FAST_OCR_PLATE_WIDTH = 75
MIN_OCR_CONFIDENCE = 0.38
MIN_PLATE_OCR_WIDTH = 55
MIN_PLATE_OCR_HEIGHT = 16
TRACKER_CONFIG = str(BASE_DIR / "custom_bytetrack.yaml") if (BASE_DIR / "custom_bytetrack.yaml").exists() else "bytetrack.yaml"
PLAYBACK_SPEEDS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

# Vehicle classes recognized by YOLO11 (COCO standard classes refined for Indian traffic)
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct UI Bounding Box Colors (BGR format)
CLASS_COLORS = {
    "car": (0, 255, 0),             # Green
    "motorcycle": (0, 165, 255),      # Orange
    "auto-rickshaw": (0, 255, 255),   # Yellow
    "bus": (255, 200, 0),            # Cyan-Blue
    "truck": (255, 0, 255),          # Magenta
    "bicycle": (128, 255, 0),        # Lime
}

# Indian State & Union Territory Registration Codes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UA", "UP", "WB", "BH"
}

# Character disambiguation dictionaries
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

# In series positions, 'I' and 'O' are not allocated in India to prevent confusion with 1 and 0.
DIGIT_TO_SERIES_CHAR = {
    "0": "D", "O": "D",
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

# Optical confusion substitution penalties (smaller = more visually similar)
VISUAL_CONFUSION_COSTS = {
    ("U", "D"): 0.10, ("D", "U"): 0.10,
    ("O", "D"): 0.10, ("D", "O"): 0.10,
    ("O", "0"): 0.05, ("0", "O"): 0.05,
    ("D", "0"): 0.05, ("0", "D"): 0.05,
    ("V", "Y"): 0.10, ("Y", "V"): 0.10,
    ("V", "U"): 0.15, ("U", "V"): 0.15,
    ("B", "8"): 0.08, ("8", "B"): 0.08,
    ("S", "5"): 0.08, ("5", "S"): 0.08,
    ("Z", "2"): 0.08, ("2", "Z"): 0.08,
    ("I", "1"): 0.05, ("1", "I"): 0.05,
    ("T", "1"): 0.12, ("1", "T"): 0.12,
    ("I", "T"): 0.12, ("T", "I"): 0.12,
    ("G", "6"): 0.10, ("6", "G"): 0.10,
    ("A", "4"): 0.10, ("4", "A"): 0.10,
    ("J", "7"): 0.12, ("7", "J"): 0.12,
    ("P", "9"): 0.15, ("9", "P"): 0.15,
    ("C", "0"): 0.12, ("0", "C"): 0.12,
    ("C", "O"): 0.10, ("O", "C"): 0.10,
    ("Q", "0"): 0.08, ("0", "Q"): 0.08,
    ("4", "2"): 0.10, ("2", "4"): 0.10,
    ("M", "N"): 0.10, ("N", "M"): 0.10,
    ("Q", "D"): 0.10, ("D", "Q"): 0.10,
    ("Q", "O"): 0.08, ("O", "Q"): 0.08,
}

# OCR state code correction mappings
STATE_PREFIX_CORRECTIONS = {
    # Uttar Pradesh (UP)
    "OP": "UP", "VP": "UP", "0P": "UP", "IP": "UP", "JP": "UP", "LP": "UP", "CP": "UP",
    "KP": "UP", "BP": "UP", "FP": "UP", "DP": "UP", "YP": "UP", "WP": "UP", "TP": "UP", "UP": "UP",
    # Delhi (DL)
    "0L": "DL", "OL": "DL", "D1": "DL", "DI": "DL", "DJ": "DL", "OI": "DL", "QL": "DL",
    "IL": "DL", "1L": "DL", "CL": "DL", "UL": "DL", "DH": "DL", "01": "DL", "O1": "DL", "DL": "DL",
    # Odisha (OD)
    "0D": "OD", "UD": "OD", "CD": "OD", "QD": "OD", "OR": "OD", "0R": "OD", "OD": "OD",
    # Maharashtra (MH)
    "M1": "MH", "MR": "MH", "M4": "MH", "NH": "MH", "MI": "MH", "MH": "MH",
    # Karnataka (KA)
    "K4": "KA", "K8": "KA", "KA": "KA",
    # Haryana (HR)
    "HB": "HR", "HA": "HR", "H8": "HR", "HP": "HP", "HR": "HR",
    # Gujarat (GJ)
    "6J": "GJ", "G1": "GJ", "GI": "GJ", "GY": "GJ", "CJ": "GJ", "GJ": "GJ",
    # Rajasthan (RJ)
    "8J": "RJ", "R1": "RJ", "RI": "RJ", "PJ": "RJ", "RY": "RJ", "RJ": "RJ",
    # Tamil Nadu (TN)
    "TM": "TN", "IN": "TN", "1N": "TN", "7N": "TN", "TN": "TN",
    # West Bengal (WB)
    "W8": "WB", "VB": "WB", "UB": "WB", "MB": "WB", "WB": "WB",
    # Telangana (TS)
    "T5": "TS", "IS": "TS", "1S": "TS", "7S": "TS", "TS": "TS",
    # Andhra Pradesh (AP)
    "A8": "AP", "4P": "AP", "AP": "AP",
    # Kerala (KL)
    "K1": "KL", "KI": "KL", "KL": "KL",
    # Punjab (PB)
    "P8": "PB", "PB": "PB",
    # Bihar (BR)
    "B8": "BR", "8R": "BR", "BR": "BR",
    # Jharkhand (JH)
    "7H": "JH", "JI": "JH", "J1": "JH", "JH": "JH",
    # Chhattisgarh (CG)
    "C6": "CG", "0G": "CG", "OG": "CG", "CG": "CG",
    # Chandigarh (CH)
    "C1": "CH", "0H": "CH", "OH": "CH", "CH": "CH",
    # Madhya Pradesh (MP)
    "NP": "MP", "M9": "MP", "MP": "MP",
    # Assam (AS)
    "A5": "AS", "4S": "AS", "AS": "AS",
    # Uttarakhand (UK)
    "UK": "UK", "UA": "UK",
    # Bharat Series (BH)
    "8H": "BH", "BH": "BH",
}


def configure_torch():
    """Optimizes PyTorch GPU execution on NVIDIA RTX GPUs."""
    if GPU_AVAILABLE:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
