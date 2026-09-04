import config
from ocr import parse_and_score_plate


class IdentityRegistry:
    """
    Assigns sequential vehicle tracking IDs (VEH-XXXXX) and unique cross-video Re-IDs (REID-XXXXX).
    Ensures that 1 physical vehicle / confirmed plate has exactly ONE permanent Re-ID.
    """
    def __init__(self):
        self.next_track_number = 1
        self.plate_to_reid = {}
        self.reid_to_plate = {}
        self.next_reid_number = 1

    def new_track_id(self):
        track_id = f"VEH-{self.next_track_number:05d}"
        self.next_track_number += 1
        return track_id

    def reid_for_plate(self, plate_text, current_reid=""):
        if not plate_text:
            return current_reid or ""

        # 1. If this exact plate is already registered to an established Re-ID (e.g. from previous video)
        if plate_text in self.plate_to_reid:
            existing_reid = self.plate_to_reid[plate_text]
            if current_reid and current_reid != existing_reid:
                old_p = self.reid_to_plate.pop(current_reid, None)
                if old_p and old_p in self.plate_to_reid:
                    del self.plate_to_reid[old_p]
            return existing_reid

        # 2. Optical confusion match (merges 4 vs 2, M vs N, Q vs D variants under the same physical vehicle)
        for existing_plate, existing_reid in list(self.plate_to_reid.items()):
            if len(existing_plate) == len(plate_text):
                cost = 0.0
                for c1, c2 in zip(existing_plate, plate_text):
                    if c1 == c2:
                        continue
                    cost += config.VISUAL_CONFUSION_COSTS.get((c1, c2), config.VISUAL_CONFUSION_COSTS.get((c2, c1), 1.0))
                if cost <= 0.15:
                    # Merge! Update existing entry to the newer/clearer plate reading
                    del self.plate_to_reid[existing_plate]
                    self.plate_to_reid[plate_text] = existing_reid
                    self.reid_to_plate[existing_reid] = plate_text
                    return existing_reid

        # 3. If the vehicle already has an Re-ID, retain it and update plate association (refinement)
        if current_reid:
            old_plate = self.reid_to_plate.get(current_reid)
            if old_plate and old_plate in self.plate_to_reid:
                del self.plate_to_reid[old_plate]
            self.plate_to_reid[plate_text] = current_reid
            self.reid_to_plate[current_reid] = plate_text
            return current_reid

        # 4. Mint brand-new Re-ID for this physical vehicle
        reid = f"REID-{self.next_reid_number:05d}"
        self.next_reid_number += 1
        self.plate_to_reid[plate_text] = reid
        self.reid_to_plate[reid] = plate_text
        return reid


class VehicleTrackerProfile:
    """
    Maintains the complete multi-frame history and temporal consensus profile
    for a tracked vehicle across the entire video.
    """
    def __init__(self, vehicle_id, video_name, initial_frame, vehicle_type, vehicle_conf):
        self.vehicle_id = vehicle_id
        self.video_name = video_name
        self.first_frame = initial_frame
        self.last_frame = initial_frame
        self.vehicle_type = vehicle_type
        self.vehicle_conf = vehicle_conf
        self.class_votes = {vehicle_type: vehicle_conf}
        self.re_id = ""
        self.plate_text = ""
        self.ocr_confidence = 0.0
        self.plate_detected = "no"
        self.best_crop = None
        self.best_sharpness = -1.0
        self.best_score = 0.0
        self.best_plate_w = 0
        self.frame_readings = []
        self._last_area = 0

    def update_vehicle_detection(self, vtype, vconf, frame_number, vehicle_box=None):
        """Dynamically updates vehicle class and monitors motion to suppress static roadside barriers."""
        self.last_frame = frame_number
        self.class_votes[vtype] = max(self.class_votes.get(vtype, 0.0), vconf)

        if vehicle_box is not None:
            cx = (vehicle_box[0] + vehicle_box[2]) // 2
            cy = (vehicle_box[1] + vehicle_box[3]) // 2
            if not hasattr(self, "centroids"):
                self.centroids = []
            self.centroids.append((cx, cy))
            if len(self.centroids) > 50:
                self.centroids.pop(0)
            # If an object sits completely still for 25+ frames with zero plate, it's static roadside clutter
            if len(self.centroids) >= 25 and self.plate_detected == "no":
                dx = max(c[0] for c in self.centroids) - min(c[0] for c in self.centroids)
                dy = max(c[1] for c in self.centroids) - min(c[1] for c in self.centroids)
                if dx < 8 and dy < 8:
                    self.is_stationary_barrier = True
        
        # In India: All passenger cars, SUVs, MUVs, sedans, and hatchbacks are 'car'
        # Only true buses (bus >= 0.45) and heavy goods trucks (truck >= 0.65 AND truck > car + 0.15) are categorized as bus/truck
        car_score = self.class_votes.get("car", 0.0)
        bus_score = self.class_votes.get("bus", 0.0)
        truck_score = self.class_votes.get("truck", 0.0)
        moto_score = self.class_votes.get("motorcycle", 0.0)
        
        if moto_score >= 0.40 and moto_score >= car_score:
            self.vehicle_type = "motorcycle"
            self.vehicle_conf = moto_score
        elif bus_score >= 0.35 and bus_score >= car_score:
            self.vehicle_type = "bus"
            self.vehicle_conf = bus_score
        elif truck_score >= 0.65 and truck_score > car_score + 0.15:
            self.vehicle_type = "truck"
            self.vehicle_conf = truck_score
        else:
            self.vehicle_type = "car"
            self.vehicle_conf = max(car_score, vconf)

    def add_reading(self, plate_text, ocr_conf, composite_score, crop, sharpness, plate_w, frame_number):
        self.plate_detected = "yes"
        self.last_frame = frame_number

        # Save highest clarity golden crop (weighted by size and sharpness)
        crop_quality = plate_w * max(sharpness, 1.0)
        current_best_quality = self.best_plate_w * max(self.best_sharpness, 1.0)
        if crop is not None and crop.size > 0 and (crop_quality > current_best_quality or self.best_crop is None):
            self.best_crop = crop.copy()
            self.best_sharpness = sharpness
            self.best_plate_w = plate_w

        # If already locked with confirmed high-confidence consensus, prevent any dilution
        if getattr(self, "is_locked", False):
            return

        if plate_text and ocr_conf >= config.MIN_OCR_CONFIDENCE:
            self.frame_readings.append({
                "text": plate_text,
                "conf": ocr_conf,
                "score": composite_score,
                "sharpness": max(sharpness, 1.0),
                "width": max(plate_w, 30),
                "frame": frame_number,
            })
            self.update_consensus()

    def update_consensus(self):
        if not self.frame_readings or getattr(self, "is_locked", False):
            return

        if len(self.frame_readings) == 1:
            r = self.frame_readings[0]
            self.plate_text = r["text"]
            self.ocr_confidence = r["conf"]
            self.best_score = r["score"]
            return

        max_w = max(r["width"] for r in self.frame_readings)
        max_sharpness = max(r["sharpness"] for r in self.frame_readings)
        max_score = max(r["score"] for r in self.frame_readings)

        # Focus voting on higher-quality observations (closer to camera and sharper)
        quality_readings = [
            r for r in self.frame_readings
            if (r["width"] >= 0.50 * max_w and r["sharpness"] >= 0.35 * max_sharpness)
        ]
        if not quality_readings:
            quality_readings = self.frame_readings

        # Whole-string candidate aggregation (prevents cross-frame character scrambling)
        string_weights = {}
        string_confs = {}

        for r in quality_readings:
            txt = r["text"]
            rel_w = r["width"] / float(max_w)
            rel_sharp = r["sharpness"] / float(max_sharpness)
            # Quadratic distance bonus: prioritize large, close-up, sharp crops
            conf_mult = 2.5 if r["conf"] >= 0.95 else 1.0
            weight = (r["score"] ** 2) * (rel_w ** 2.2) * (rel_sharp ** 1.0) * conf_mult
            string_weights[txt] = string_weights.get(txt, 0.0) + weight
            string_confs[txt] = max(string_confs.get(txt, 0.0), r["conf"])

        # Pick highest-weighted authentic plate string (close-up, high-confidence readings dominate)
        best_str = max(string_weights.items(), key=lambda item: item[1])[0]
        formatted, bonus, _ = parse_and_score_plate(best_str)

        if formatted and bonus > 0.0:
            self.plate_text = formatted
            self.ocr_confidence = string_confs.get(best_str, 0.85)
            self.best_score = max_score
        else:
            best_r = max(quality_readings, key=lambda r: (r["score"], r["width"], r["sharpness"]))
            self.plate_text = best_r["text"]
            self.ocr_confidence = best_r["conf"]
            self.best_score = best_r["score"]
            self.plate_text = best_r["text"]
            self.ocr_confidence = best_r["conf"]
            self.best_score = best_r["score"]

    def to_record_dict(self):
        return {
            "vehicle_id": self.vehicle_id,
            "re_id": self.re_id,
            "plate_text": self.plate_text,
            "ocr_confidence": f"{self.ocr_confidence:.2f}" if self.plate_text else "",
            "video": self.video_name,
            "frame": self.last_frame,
            "vehicle_type": self.vehicle_type,
            "vehicle_confidence": f"{self.vehicle_conf:.2f}",
            "plate_detected": self.plate_detected,
        }
