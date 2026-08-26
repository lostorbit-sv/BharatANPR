from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VIDEO_DIR = BASE_DIR / "video"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# COCO class ids used by YOLO for common road vehicles.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    model = YOLO(str(MODEL_PATH))
    video_paths = sorted(
        path for path in VIDEO_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        raise FileNotFoundError(f"No videos found in: {VIDEO_DIR}")

    for video_path in video_paths:
        play_video(model, video_path)

    cv2.destroyAllWindows()


def play_video(model, video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Skipping video because it could not be opened: {video_path}")
        return

    print(f"Playing: {video_path.name}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.25, classes=list(VEHICLE_CLASSES), verbose=False)
        vehicle_count = 0

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{VEHICLE_CLASSES.get(class_id, model.names[class_id])} {confidence:.2f}"

                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            video_path.name,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("YOLO Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        if cv2.getWindowProperty("YOLO Vehicle Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()


if __name__ == "__main__":
    main()
