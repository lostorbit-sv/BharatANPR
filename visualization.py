import cv2
import numpy as np
import config


def find_video_paths():
    """Scans configured video directories for playable video clips."""
    video_paths = []
    for directory in config.VIDEO_DIRECTORIES:
        if directory.is_dir():
            video_paths.extend(
                path for path in directory.iterdir() if path.suffix.lower() in config.VIDEO_EXTENSIONS
            )
    return sorted(set(video_paths), key=lambda path: (path.parent.name.lower(), path.name.lower()))


def choose_videos(video_paths):
    """Interactive GUI window allowing user to browse and select videos."""
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
            f"Marked: {len(marked_indexes)} (Press ENTER to play, SPACE to select, A for all)",
            (30, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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
        if raw_key in (2490368, 82):  # Up arrow
            selected_index = max(0, selected_index - 1)
        if raw_key in (2621440, 84):  # Down arrow
            selected_index = min(len(video_paths) - 1, selected_index + 1)
        if key in (27, ord("q")) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            return []


def draw_vehicle(frame, vehicle_box, record, vehicle_type, confidence):
    """Draws color-coded vehicle bounding box with ID, type, and recognized plate."""
    x1, y1, x2, y2 = vehicle_box
    color = config.CLASS_COLORS.get(vehicle_type, (0, 255, 0))
    identity = record["re_id"] or record["vehicle_id"]
    plate_info = f" | {record['plate_text']}" if record.get("plate_text") else ""
    label = f"{identity} | {vehicle_type} {confidence:.2f}{plate_info}"
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
    """Draws red bounding box and text label for the detected license plate."""
    if len(plate_box) >= 5:
        x1, y1, x2, y2, confidence = plate_box[:5]
    else:
        x1, y1, x2, y2 = plate_box[:4]
        confidence = 1.0
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
    """Draws top HUD status overlay with hardware, FPS, speed, and video name."""
    device_label = "GPU" if config.GPU_AVAILABLE else "CPU"
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
