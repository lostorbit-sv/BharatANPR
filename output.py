import csv
import time
import cv2
import config


def init_output_dirs():
    """Ensures output directories exist."""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.PLATE_CROP_DIR.mkdir(exist_ok=True)


def open_csv_writer(base_filename="unique_vehicles.csv"):
    """
    Opens a CSV writer for the results file. If the file is locked (e.g. open in Excel),
    gracefully falls back to a timestamped filename so the program never crashes.
    """
    init_output_dirs()
    target_path = config.OUTPUT_DIR / base_filename
    
    try:
        file_handle = target_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = target_path.stem
        fallback_path = config.OUTPUT_DIR / f"{stem}_{timestamp}.csv"
        print(f"\n[WARNING] '{target_path.name}' is currently locked (e.g. open in Excel).")
        print(f"[INFO] Saving results to fallback file: '{fallback_path.name}' instead.\n")
        file_handle = fallback_path.open("w", newline="", encoding="utf-8")
        target_path = fallback_path

    writer = csv.DictWriter(
        file_handle,
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
    return file_handle, writer, target_path


def save_plate_crop(plate_crop, re_id):
    """Saves the highest-sharpness golden crop for a vehicle plate."""
    if plate_crop is not None and plate_crop.size and re_id:
        crop_path = config.PLATE_CROP_DIR / f"{re_id}.jpg"
        cv2.imwrite(str(crop_path), plate_crop)


def log_plate_detection(action_type, vehicle_id, re_id, plate_text, conf, v_type, frame_num, video_name, frame_count=1):
    """Streams live formatted detection and refinement notifications to CMD / terminal."""
    timestamp = time.strftime("%H:%M:%S")
    re_id_str = re_id if re_id else "PENDING"
    if action_type == "DETECTED":
        print(f"[{timestamp}] [DETECTED] {vehicle_id} -> {re_id_str:<10} | Plate: {plate_text:<12} (Conf: {conf:.2f}) | {v_type:<14} | Frame: {frame_num} | {video_name}")
    elif action_type == "REFINED":
        print(f"[{timestamp}] [REFINED]  {vehicle_id} -> {re_id_str:<10} | Plate: {plate_text:<12} (Conf: {conf:.2f}, Frames: {frame_count:02d}) | Golden Crop Saved")


def print_video_summary(video_name, total_tracked, plates_read, unique_reids):
    """Prints a summary table to CMD when a video completes processing."""
    read_pct = (plates_read / max(1, total_tracked)) * 100.0
    print("\n" + "=" * 80)
    print(f"  VIDEO SUMMARY: {video_name}")
    print(f"  Total Vehicles Tracked     : {total_tracked}")
    print(f"  Plates Successfully Read   : {plates_read} ({read_pct:.1f}%)")
    print(f"  Unique Re-IDs Registered   : {unique_reids}")
    print(f"  Saved High-Res Golden Crops: outputs/plate_crops/")
    print(f"  CSV Report Updated         : outputs/unique_vehicles.csv")
    print("=" * 80 + "\n")
