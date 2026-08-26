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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv",}

GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if GPU_AVAILABLE else "cpu"

# The stronger model is used on the RTX GPU. The CPU fallback keeps playback usable.
VEHICLE_MODEL_PATH = BASE_DIR / ("yolo11s.pt" if GPU_AVAILABLE else "yolo11n.pt")
VEHICLE_IMAGE_SIZE = 704 if GPU_AVAILABLE else 640
PLATE_IMAGE_SIZE = 640
VEHICLE_CONFIDENCE = 0.30
PLATE_CONFIDENCE = 0.25
PLATE_DETECTION_INTERVAL = 4
OCR_RETRY_INTERVAL = 30
MIN_OCR_CONFIDENCE = 0.45

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
        key = cv2.waitKey(30) & 0xFF
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
        if key in (81, 82):  # Left/up arrow
            selected_index = max(0, selected_index - 1)
        if key in (83, 84):  # Right/down arrow
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
    next_ocr_frame = {}
    cached_plate_boxes = []
    displayed_fps = 0.0
    playback_action = "next"
    paused = False

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
                    if plates_are_fresh and frame_number >= next_ocr_frame.get(vehicle_id, 0):
                        plate_text, ocr_confidence, plate_crop = read_plate_text(
                            ocr_reader, frame, matching_plate
                        )
                        next_ocr_frame[vehicle_id] = frame_number + OCR_RETRY_INTERVAL

                        if plate_text and ocr_confidence >= MIN_OCR_CONFIDENCE:
                            record["plate_text"] = plate_text
                            record["ocr_confidence"] = f"{ocr_confidence:.2f}"
                            record["re_id"] = registry.reid_for_plate(plate_text)
                            save_plate_crop(plate_crop, record["re_id"])

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
        )
        cv2.imshow("Vehicle, Plate OCR, and Re-ID", frame)

        key = cv2.waitKey(1) & 0xFF
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
        if cv2.getWindowProperty("Vehicle, Plate OCR, and Re-ID", cv2.WND_PROP_VISIBLE) < 1:
            playback_action = "quit"
            break

    cap.release()
    writer.writerows(vehicle_records.values())
    return playback_action


def reset_tracker(vehicle_model):
    predictor = getattr(vehicle_model, "predictor", None)
    if predictor is None or not hasattr(predictor, "trackers"):
        return

    for tracker in predictor.trackers:
        tracker.reset()


def get_plate_boxes(plate_results):
    boxes = []
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
    return boxes


def find_plate_for_vehicle(vehicle_box, plate_boxes):
    vx1, vy1, vx2, vy2 = vehicle_box
    matching_plate = None
    best_confidence = -1.0

    for px1, py1, px2, py2, confidence in plate_boxes:
        plate_center_x = (px1 + px2) // 2
        plate_center_y = (py1 + py2) // 2
        is_inside_vehicle = vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2

        if is_inside_vehicle and confidence > best_confidence:
            matching_plate = (px1, py1, px2, py2, confidence)
            best_confidence = confidence

    return matching_plate


def read_plate_text(ocr_reader, frame, plate_box):
    x1, y1, x2, y2, _ = plate_box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    plate_crop = frame[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return "", 0.0, plate_crop

    prepared_crop = prepare_plate_for_ocr(plate_crop)
    detections = ocr_reader.readtext(
        prepared_crop,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    candidates = []
    for _, text, confidence in detections:
        cleaned_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if 4 <= len(cleaned_text) <= 12:
            candidates.append((cleaned_text, float(confidence)))

    if not candidates:
        return "", 0.0, plate_crop
    return max(candidates, key=lambda candidate: candidate[1]) + (plate_crop,)


def prepare_plate_for_ocr(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enlarged)


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


def draw_status(frame, video_name, active_count, unique_count, fps):
    device_label = "GPU" if GPU_AVAILABLE else "CPU"
    cv2.putText(
        frame,
        f"{device_label} | {fps:.1f} FPS | Active: {active_count} | Tracks: {unique_count}",
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
