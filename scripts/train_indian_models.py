import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\amita\OneDrive\Desktop\anpr_project")
V102_DIR = Path(r"C:\Users\amita\OneDrive\Desktop\anpr v1.0.2")

def train_vehicle_model(epochs=20, batch=8, imgsz=640):
    print("=" * 80)
    print("PHASE 1: TRAINING INDIAN VEHICLE YOLO11 (AUTO-RICKSHAW, BIKE, CAR, BUS, TRUCK)")
    print("=" * 80)
    
    data_yaml = BASE_DIR / "dataset" / "vehicles_indian" / "data.yaml"
    base_model = BASE_DIR / "yolo11s.pt"
    if not base_model.exists():
        base_model = "yolo11s.pt"
        
    model = YOLO(str(base_model))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        amp=True,
        workers=0,
        plots=False,
        project=str(BASE_DIR / "runs" / "train_vehicles"),
        name="indian_vehicles",
        exist_ok=True,
        verbose=True,
    )
    
    # Locate best weights
    best_weights = BASE_DIR / "runs" / "train_vehicles" / "indian_vehicles" / "weights" / "best.pt"
    if best_weights.exists():
        target_proj = BASE_DIR / "indian_vehicles_yolo11.pt"
        shutil.copy2(best_weights, target_proj)
        print(f"[SUCCESS] Saved Indian Vehicle Model to: {target_proj}")
        if V102_DIR.exists():
            target_v102 = V102_DIR / "indian_vehicles_yolo11.pt"
            shutil.copy2(best_weights, target_v102)
            print(f"[SUCCESS] Synced Indian Vehicle Model to: {target_v102}")
    return results


def train_plate_model(epochs=20, batch=16, imgsz=640):
    print("=" * 80)
    print("PHASE 2: TRAINING UNIVERSAL INDIAN LICENSE PLATE YOLO11")
    print("=" * 80)
    
    data_yaml = BASE_DIR / "dataset" / "plates_indian" / "data.yaml"
    base_plate = BASE_DIR / "license_plate_detector.pt"
    if not base_plate.exists():
        base_plate = BASE_DIR / "yolo11s.pt"
        
    model = YOLO(str(base_plate))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        amp=True,
        workers=0,
        cache="ram",
        plots=False,
        project=str(BASE_DIR / "runs" / "train_plates"),
        name="indian_plates",
        exist_ok=True,
        verbose=True,
    )
    
    best_weights = BASE_DIR / "runs" / "train_plates" / "indian_plates" / "weights" / "best.pt"
    if best_weights.exists():
        target_proj = BASE_DIR / "indian_plate_yolo11.pt"
        shutil.copy2(best_weights, target_proj)
        print(f"[SUCCESS] Saved Indian Plate Model to: {target_proj}")
        if V102_DIR.exists():
            target_v102 = V102_DIR / "indian_plate_yolo11.pt"
            shutil.copy2(best_weights, target_v102)
            print(f"[SUCCESS] Synced Indian Plate Model to: {target_v102}")
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("vehicles", "both"):
        train_vehicle_model()
    if mode in ("plates", "both"):
        train_plate_model()
