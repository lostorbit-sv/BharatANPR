import os
import json
import tarfile
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

random.seed(42)

BASE_DIR = Path(r"C:\Users\amita\OneDrive\Desktop\anpr_project")
DATASET_DIR = BASE_DIR / "dataset"
VEHICLES_DIR = DATASET_DIR / "vehicles_indian"
PLATES_DIR = DATASET_DIR / "plates_indian"

# Class mappings for Indian Vehicles
VEHICLE_CLASS_MAP = {
    "car": 0,
    "motorcycle": 1,
    "rider": 1,
    "autorickshaw": 2,
    "bus": 3,
    "truck": 4,
    "bicycle": 5,
}

VEHICLE_NAMES = ["car", "motorcycle", "autorickshaw", "bus", "truck", "bicycle"]


def prepare_vehicles_from_delhi_shard(tar_path):
    print("=" * 70)
    print("EXTRACTING & CONVERTING DELHI NCR ROAD DATASET (YOLO FORMAT)")
    print(f"Source Shard: {tar_path}")
    print("=" * 70)

    train_img_dir = VEHICLES_DIR / "images" / "train"
    val_img_dir = VEHICLES_DIR / "images" / "val"
    train_lbl_dir = VEHICLES_DIR / "labels" / "train"
    val_lbl_dir = VEHICLES_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        json_names = [n for n in names if n.endswith(".json")]
        print(f"Total annotated scenes in shard: {len(json_names)}")

        processed = 0
        total_boxes = 0

        for jn in json_names:
            base_id = jn[:-5]
            img_name = f"{base_id}.jpg"
            if img_name not in names:
                continue

            # Read JSON annotation
            f_json = tar.extractfile(jn)
            data = json.load(f_json)
            labels = data.get("labels", [])

            # Extract image to memory to get dimensions
            f_img = tar.extractfile(img_name)
            img_bytes = f_img.read()
            img_np = cv2.imdecode(__import__("numpy").frombuffer(img_bytes, dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
            if img_np is None:
                continue
            h, w = img_np.shape[:2]

            yolo_lines = []
            for item in labels:
                cat = item.get("category", "").lower()
                if cat not in VEHICLE_CLASS_MAP:
                    continue

                cls_id = VEHICLE_CLASS_MAP[cat]
                box = item.get("box2d")
                if not box:
                    continue

                x1 = max(0.0, float(box["x1"]))
                y1 = max(0.0, float(box["y1"]))
                x2 = min(float(w), float(box["x2"]))
                y2 = min(float(h), float(box["y2"]))

                bw = x2 - x1
                bh = y2 - y1
                # Filter out microscopic noise
                if bw < 14 or bh < 14:
                    continue

                x_c = (x1 + x2) / 2.0 / w
                y_c = (y1 + y2) / 2.0 / h
                nw = bw / w
                nh = bh / h

                yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")
                total_boxes += 1

            if not yolo_lines:
                continue

            # 85% train, 15% validation split
            is_val = random.random() < 0.15
            dest_img = (val_img_dir if is_val else train_img_dir) / f"{base_id}.jpg"
            dest_lbl = (val_lbl_dir if is_val else train_lbl_dir) / f"{base_id}.txt"

            with open(dest_img, "wb") as f_out:
                f_out.write(img_bytes)

            with open(dest_lbl, "w") as f_out:
                f_out.write("\n".join(yolo_lines) + "\n")

            processed += 1

    print(f"Vehicles Dataset Ready: {processed} images saved with {total_boxes} vehicle boxes!")

    # Write data.yaml for YOLO training
    yaml_content = f"""path: {str(VEHICLES_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(VEHICLE_NAMES):
        yaml_content += f"  {i}: {name}\n"

    with open(VEHICLES_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Saved: {VEHICLES_DIR / 'data.yaml'}")


def prepare_plates_from_coco_zip(zip_path):
    print("=" * 70)
    print("EXTRACTING & CONVERTING LICENSE PLATES DATASET (YOLO FORMAT)")
    print(f"Source Zip: {zip_path}")
    print("=" * 70)

    train_img_dir = PLATES_DIR / "images" / "train"
    val_img_dir = PLATES_DIR / "images" / "val"
    train_lbl_dir = PLATES_DIR / "labels" / "train"
    val_lbl_dir = PLATES_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        coco_name = "_annotations.coco.json"
        if coco_name not in z.namelist():
            print("Error: _annotations.coco.json not found in zip!")
            return

        with z.open(coco_name) as f:
            coco = json.load(f)

        img_id_to_file = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]}
        img_to_boxes = {}

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            bbox = ann["bbox"] # [x, y, width, height]
            if img_id not in img_to_boxes:
                img_to_boxes[img_id] = []
            img_to_boxes[img_id].append(bbox)

        processed = 0
        total_plates = 0

        for img_id, (fname, img_w, img_h) in img_id_to_file.items():
            if fname not in z.namelist() or img_id not in img_to_boxes:
                continue

            yolo_lines = []
            for (bx, by, bw, bh) in img_to_boxes[img_id]:
                if bw < 8 or bh < 6:
                    continue
                x_c = (bx + bw / 2.0) / img_w
                y_c = (by + bh / 2.0) / img_h
                nw = bw / float(img_w)
                nh = bh / float(img_h)
                # Class 0: license_plate
                yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")
                total_plates += 1

            if not yolo_lines:
                continue

            img_bytes = z.read(fname)
            is_val = random.random() < 0.15
            dest_img = (val_img_dir if is_val else train_img_dir) / fname
            dest_lbl = (val_lbl_dir if is_val else train_lbl_dir) / (Path(fname).stem + ".txt")

            with open(dest_img, "wb") as f_out:
                f_out.write(img_bytes)

            with open(dest_lbl, "w") as f_out:
                f_out.write("\n".join(yolo_lines) + "\n")

            processed += 1

    print(f"License Plates Dataset Ready: {processed} images saved with {total_plates} plate boxes!")

    yaml_content = f"""path: {str(PLATES_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

names:
  0: license_plate
"""
    with open(PLATES_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Saved: {PLATES_DIR / 'data.yaml'}")


if __name__ == "__main__":
    shard0 = Path(r"C:\Users\amita\.cache\huggingface\hub\datasets--thirdeyelabs--indian-road-dataset\snapshots\69e29b09a26d21036c4fc6b0d7022b705a350cb2\data\train-00000-of-00646.tar")
    if shard0.exists():
        prepare_vehicles_from_delhi_shard(shard0)
