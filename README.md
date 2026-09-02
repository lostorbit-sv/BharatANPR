# ANPR Vehicle Detection and Tracking

A Windows desktop project for detecting road vehicles, locating license plates, reading plate text with OCR, and assigning tracking and re-identification IDs across local videos.

## Features

- Detects cars, motorcycles, buses, and trucks with YOLO11.
- Uses the NVIDIA GPU automatically when CUDA is available.
- Locates license plates with a dedicated YOLO plate detector.
- Reads detected plate text with EasyOCR.
- Assigns a temporary vehicle track ID such as `VEH-00001`.
- Assigns a Re-ID such as `REID-00001` when a plate is read with sufficient confidence.
- Matches the same accepted plate across videos processed in the same run.
- Provides a compact video selector and keyboard playback controls.
- Exports vehicle and plate results to CSV.

## Requirements

- Windows 10 or Windows 11
- Python 3.13
- NVIDIA GPU recommended. The current project is configured for CUDA 12.8 and was tested with an NVIDIA GeForce RTX 3050 Laptop GPU.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the project packages:

```powershell
pip install -r requirements.txt
```

EasyOCR may install `opencv-python-headless`, which cannot open the video-player window. Restore the normal Windows OpenCV build after installing the requirements:

```powershell
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python
```

## Model Files

Place these files in the project root:

| File | Purpose | Download |
| --- | --- | --- |
| `yolo11s.pt` | Vehicle detection on GPU | [Ultralytics YOLO11s weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| `yolo11n.pt` | CPU fallback vehicle detection | Download automatically through Ultralytics, or use the existing project copy |
| `license_plate_detector.pt` | License plate detection | [License plate detector weights](https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt?download=true) |

Rename the downloaded license-plate model to `license_plate_detector.pt`.

## Folder Structure

```text
anpr_project/
├── main.py
├── requirements.txt
├── yolo11s.pt
├── yolo11n.pt
├── license_plate_detector.pt
├── videos/
│   ├── traffic.mp4
│   └── other_video.avi
└── outputs/
```

The program searches for videos in `videos`, `video`, and `video_1`. Supported formats are `.mp4`, `.avi`, `.mov`, `.mkv`, and `.wmv`.

## Run

```powershell
python main.py
```

The first window is the video selector:

- Up and Down arrow keys: move through videos.
- `Space`: mark or unmark a video.
- `Enter`: play marked videos, or the highlighted video.
- `A`: play all videos.
- `Q` or `Esc`: close the selector.

During detection playback:

- `Space`: pause or resume.
- `N`: next video.
- `P`: previous video.
- `-`: decrease playback speed.
- `+` or `=`: increase playback speed.
- `Q`: exit.

## Output

The application writes:

- `outputs/unique_vehicles.csv`: vehicle tracks, vehicle type, plate detection status, plate text, OCR confidence, and Re-ID.
- `outputs/plate_crops/`: plate images for accepted OCR readings.

## Notes

- The OCR confidence threshold is intentionally conservative. A weak plate reading is not used to create a Re-ID, which avoids false vehicle matches.
- Re-IDs are matched by accepted plate text during one program run. A persistent database is the next step for retaining vehicle identities across separate runs.
- OCR quality depends heavily on the plate size, angle, blur, lighting, and whether the detection model was trained for the country or plate style in the source videos.
