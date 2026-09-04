import cv2
import numpy as np
import config


def validate_project_files():
    """Ensures all required model weights and directories exist before running."""
    missing = []
    if not config.PLATE_MODEL_PATH.exists():
        missing.append(f"Plate model missing: {config.PLATE_MODEL_PATH}")
    if not config.VEHICLE_MODEL_PATH.exists():
        missing.append(f"Vehicle model missing: {config.VEHICLE_MODEL_PATH}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def reset_tracker(model):
    """Resets ByteTrack persistent IDs between different video files or on seeks."""
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        if hasattr(predictor, "trackers"):
            delattr(predictor, "trackers")
        if hasattr(predictor, "vid_path"):
            delattr(predictor, "vid_path")


def is_valid_plate_box(plate_box):
    """
    Validates license plate aspect ratio and minimum size.
    Rejects bumper skirts, road textures, and slit noise.
    """
    x1, y1, x2, y2 = plate_box[:4]
    w = x2 - x1
    h = y2 - y1
    if w < 35 or h < 12:
        return False
    aspect = w / float(max(1, h))
    # Indian plates: 1.4 (square commercial/2-line) to 5.2 (standard rectangular)
    if aspect < 1.35 or aspect > 5.5:
        return False
    return True


def get_plate_boxes(plate_results):
    """
    Extracts valid license plate bounding boxes (x1, y1, x2, y2, confidence).
    Filters out invalid aspect ratios and low-confidence artifacts.
    """
    plate_boxes = []
    for result in plate_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < config.PLATE_CONFIDENCE:
                continue
            bx = tuple(map(int, box.xyxy[0])) + (conf,)
            if is_valid_plate_box(bx):
                plate_boxes.append(bx)
    return plate_boxes


def is_traffic_barricade(crop):
    """
    Detects roadside metal barricades/barriers with yellow skeletal pipe frames
    (such as police traffic barricades).
    """
    if crop is None or crop.size < 400:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Bright yellow/orange barricade pipes: Hue: 15 to 38, Sat >= 90, Val >= 90
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 90)
    yp = float(np.mean(yellow) * 100)

    # Skeletal wireframe: 4.5% to 28% yellow pipes with open/transparent background
    # (Solid yellow cabs have > 40% yellow body panels; normal cars have < 2% yellow)
    if 4.5 <= yp <= 28.0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        opened = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        opened_pct = float(np.mean(opened) * 100)
        # If open morphology wipes out most yellow, it consists of thin hollow skeletal pipes!
        if opened_pct < 0.40 * yp or yp < 15.0:
            return True
    return False


def is_valid_vehicle_detection(vehicle_box, class_name, confidence, frame_shape, frame=None):
    """
    Filters out false positive vehicle detections caused by bridge railings,
    vertical poles, road barriers, traffic barricades, or horizontal slit reflections.
    """
    x1, y1, x2, y2 = vehicle_box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = h / float(w)
    frame_h, frame_w = frame_shape[:2]

    # 1. Reject extreme vertical aspect ratios (railings/poles)
    if aspect > 2.2 or (aspect > 1.7 and h > int(frame_h * 0.65)):
        return False

    # 2. Reject horizontal slit noise (aspect < 0.20)
    if aspect < 0.20:
        return False

    # 3. Floor confidence to reject stray noise while retaining real vehicles
    if confidence < 0.18:
        return False

    # 4. Reject skeletal yellow traffic barricades
    if frame is not None and class_name in ("car", "truck"):
        crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
        if is_traffic_barricade(crop):
            return False

    return True


def is_auto_rickshaw(crop):
    """
    Detects Indian auto-rickshaws (Tuk-Tuks: Bajaj RE, Piaggio Ape, TVS King, E-rickshaws).
    Characterized by green body with yellow/white roof, black with yellow roof,
    and upright 3-wheeler chassis profile.
    """
    if crop is None or crop.size < 200:
        return False
    h, w = crop.shape[:2]
    # Buses and heavy transport trucks are massive; auto-rickshaws are compact
    if w > 420 or h > 320:
        return False
    aspect = h / float(max(1, w))
    if not (0.65 <= aspect <= 1.55):
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Upper half: roof/canopy (yellow or fabric)
    roof = hsv[:int(h * 0.55), :]
    yellow_roof = (roof[:, :, 0] >= 12) & (roof[:, :, 0] <= 42) & (roof[:, :, 1] >= 40) & (roof[:, :, 2] >= 50)

    # Lower half: body (green, dark, or yellow)
    lower = hsv[int(h * 0.35):, :]
    green_body = (lower[:, :, 0] >= 35) & (lower[:, :, 0] <= 90) & (lower[:, :, 1] >= 35) & (lower[:, :, 2] >= 35)

    y_pct = float(np.mean(yellow_roof) * 100)
    g_pct = float(np.mean(green_body) * 100)

    # Signature: yellow canopy + green body OR prominent yellow canopy on upright chassis
    if (y_pct >= 3.5 and g_pct >= 3.0) or (y_pct >= 8.0 and aspect >= 0.75):
        return True
    return False


def refine_vehicle_type(frame, vehicle_box, base_class_id):
    """Maps YOLO class ID to vehicle type (car, bus, truck, motorcycle, auto-rickshaw, bicycle)."""
    base_name = config.VEHICLE_CLASSES.get(base_class_id, "car")
    x1, y1, x2, y2 = vehicle_box
    h, w = y2 - y1, x2 - x1
    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

    # Distinguish 3-wheelers (auto-rickshaws / e-rickshaws)
    if crop.size > 200 and is_auto_rickshaw(crop):
        return "auto-rickshaw"

    if base_name == "truck":
        # In India, passenger SUVs, MUVs, and utility vehicles (Bolero, Scorpio, Safari, Creta, WagonR)
        # get predicted as COCO 'truck' by standard YOLO weights.
        # Map all standard passenger silhouettes to 'car'. Only multi-axle heavy commercial vehicles stay 'truck'.
        if (h / float(max(1, w))) < 1.05 and h < int(frame.shape[0] * 0.88):
            return "car"

    return base_name


def find_plate_for_vehicle(vehicle_box, plate_boxes):
    """
    Finds the highest-confidence license plate detection located at the bumper or tailgate of a vehicle.
    Enforces geometric priors: plates are mounted in the lower 65% of the vehicle and centered horizontally.
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    vh = vy2 - vy1
    vw = vx2 - vx1
    vc_x = (vx1 + vx2) // 2

    # A license plate is physically mounted on the bumper / lower fascia (never upper grille or roof)
    min_py = vy1 + int(vh * 0.28)
    max_py = vy2 + int(vh * 0.15)
    evx1 = vx1 - int(vw * 0.05)
    evx2 = vx2 + int(vw * 0.05)

    matching_plate = None
    best_confidence = -1.0

    for px1, py1, px2, py2, confidence in plate_boxes:
        plate_center_x = (px1 + px2) // 2
        plate_center_y = (py1 + py2) // 2

        # Reject false detections on upper radiator grille, hood, or windshield
        if plate_center_y < min_py or plate_center_y > max_py:
            continue

        # Plate must be horizontally centered on the vehicle body (within 40% of center)
        if abs(plate_center_x - vc_x) > int(vw * 0.40):
            continue

        is_inside_vehicle = evx1 <= plate_center_x <= evx2
        if is_inside_vehicle and confidence > best_confidence:
            matching_plate = (px1, py1, px2, py2, confidence)
            best_confidence = confidence

    return matching_plate


def detect_commercial_yellow_plate(frame, vehicle_box):
    """
    Fallback bumper detector for yellow commercial license plates
    (common on buses, trucks, taxis, and commercial vehicles).
    Targets the lower bumper area below grilles/hood ornaments.
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    vh = vy2 - vy1
    vw = vx2 - vx1
    # Target lower 35% of vehicle (bumper area below grille)
    cy1 = vy1 + int(vh * 0.65)
    cy2 = min(frame.shape[0], vy2 + int(vh * 0.12))
    cx1 = max(0, vx1 + int(vw * 0.08))
    cx2 = min(frame.shape[1], vx2 - int(vw * 0.08))

    bumper = frame[cy1:cy2, cx1:cx2]
    if bumper.size < 400:
        return None

    hsv = cv2.cvtColor(bumper, cv2.COLOR_BGR2HSV)
    # Commercial plate yellow: H: 15 to 36, S >= 85, V >= 85
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 36) & (hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 85)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    best_y = -1

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 35 or bh < 15 or bw > int(vw * 0.70) or bh > int(vh * 0.35):
            continue
        aspect = bw / float(max(1, bh))
        if 1.20 <= aspect <= 4.2:
            patch = bumper[y:y + bh, x:x + bw]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            std = float(np.std(gray))
            if std > 18:  # Has dark text contrast on yellow background
                # Lower on the bumper (higher Y) is the true plate position
                if y > best_y:
                    best_y = y
                    best_candidate = (cx1 + x, cy1 + y, cx1 + x + bw, cy1 + y + bh, 0.75)

    return best_candidate


def detect_plates_dual_scale(plate_model, frame, vehicle_box, cached_plate_boxes, plates_are_fresh):
    """
    Dual-scale plate detector: Full frame matching first, followed by localized zoom on vehicle front/bumper,
    with an adaptive commercial plate locator for buses and trucks.
    """
    matching_plate = find_plate_for_vehicle(vehicle_box, cached_plate_boxes)
    if matching_plate is None and plates_are_fresh:
        vx1, vy1, vx2, vy2 = vehicle_box
        vw, vh = vx2 - vx1, vy2 - vy1
        if vw >= 45 and vh >= 45:
            # Crop lower 55% of vehicle (where bumper and plates are mounted, avoiding upper grille)
            cy1 = vy1 + int(vh * 0.40)
            cy2 = min(frame.shape[0], vy2 + int(vh * 0.08))
            cx1 = max(0, vx1 - int(vw * 0.05))
            cx2 = min(frame.shape[1], vx2 + int(vw * 0.05))
            v_crop = frame[cy1:cy2, cx1:cx2]
            if v_crop.size > 200:
                v_results = plate_model(
                    v_crop,
                    conf=max(0.35, config.PLATE_CONFIDENCE),
                    imgsz=config.PLATE_IMAGE_SIZE,
                    device=config.DEVICE,
                    verbose=False,
                )
                v_boxes = get_plate_boxes(v_results)
                if v_boxes:
                    best_vbox = max(v_boxes, key=lambda b: b[4])
                    px1, py1, px2, py2, pconf = best_vbox
                    matching_plate = (cx1 + px1, cy1 + py1, cx1 + px2, cy1 + py2, pconf)

    # Fallback for commercial yellow plates (buses, trucks, cabs) if YOLO missed it
    if matching_plate is None:
        matching_plate = detect_commercial_yellow_plate(frame, vehicle_box)

    return matching_plate
