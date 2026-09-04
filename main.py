import csv
import time
import cv2
from ultralytics import YOLO

import config
from ocr import create_ocr_reader, create_parseq_model, create_lprnet_model, read_plate_text, CRNN_BiLSTM
from detection import (
    validate_project_files,
    reset_tracker,
    get_plate_boxes,
    is_valid_vehicle_detection,
    refine_vehicle_type,
    detect_plates_dual_scale,
)
from tracker import IdentityRegistry, VehicleTrackerProfile
from visualization import (
    find_video_paths,
    choose_videos,
    draw_vehicle,
    draw_plate,
    draw_status,
)
from output import (
    init_output_dirs,
    open_csv_writer,
    save_plate_crop,
    log_plate_detection,
    print_video_summary,
)


def play_video(
    vehicle_model,
    plate_model,
    ocr_reader,
    lpr_model,
    video_path,
    writer,
    registry,
    video_index,
    video_count,
):
    """
    Plays a video file with Ensemble ANPR (LPRNet + EasyOCR),
    accurate vehicle tracking, and temporal consensus.
    """
    reset_tracker(vehicle_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping video because it could not be opened: {video_path.name}")
        return "next"

    print(f"\n[{time.strftime('%H:%M:%S')}] >>> Playing ({video_index + 1}/{video_count}): {video_path.name}")
    frame_number = 0
    tracker_to_vehicle_id = {}
    vehicle_profiles = {}
    next_ocr_frame = {}
    cached_plate_boxes = []
    displayed_fps = 0.0
    playback_action = "next"
    paused = False
    playback_speed_index = config.PLAYBACK_SPEEDS.index(1.00)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek_step_frames = max(1, int(round(source_fps)))
    active_vehicle_count = 0

    def do_seek(delta_frames):
        """Cleans state and repositions video so old vehicle IDs never carry over."""
        curr_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_pos = max(0, min(total_frames - 1, curr_pos + delta_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        reset_tracker(vehicle_model)
        tracker_to_vehicle_id.clear()
        next_ocr_frame.clear()
        cached_plate_boxes.clear()
        return target_pos

    while True:
        if paused:
            raw_key = cv2.waitKeyEx(30)
            key = raw_key & 0xFF
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
            elif raw_key in (2424832, 81, 65361, ord(",")):  # Left Arrow -> Seek -1s
                target_pos = do_seek(-seek_step_frames)
                frame_number = target_pos
                ret, frame = cap.read()
                if ret:
                    display_frame = frame.copy()
                    v_res = vehicle_model.track(
                        frame, persist=True, tracker=config.TRACKER_CONFIG,
                        conf=config.VEHICLE_CONFIDENCE, classes=list(config.VEHICLE_CLASSES),
                        imgsz=config.VEHICLE_IMAGE_SIZE, device=config.DEVICE,
                        agnostic_nms=True, iou=0.45, verbose=False,
                    )
                    p_res = plate_model(frame, conf=config.PLATE_CONFIDENCE, imgsz=config.PLATE_IMAGE_SIZE, device=config.DEVICE, verbose=False)
                    cur_plates = get_plate_boxes(p_res)
                    p_active = 0
                    for result in v_res:
                        for box in result.boxes:
                            if box.id is None:
                                continue
                            c_id = int(box.cls[0])
                            v_box = tuple(map(int, box.xyxy[0]))
                            v_type = refine_vehicle_type(frame, v_box, c_id)
                            v_conf = float(box.conf[0])
                            if not is_valid_vehicle_detection(v_box, v_type, v_conf, frame.shape, frame):
                                continue
                            t_id = int(box.id[0])
                            if t_id not in tracker_to_vehicle_id:
                                tracker_to_vehicle_id[t_id] = registry.new_track_id()
                            v_id = tracker_to_vehicle_id[t_id]
                            if v_id not in vehicle_profiles:
                                vehicle_profiles[v_id] = VehicleTrackerProfile(v_id, video_path.name, frame_number, v_type, v_conf)
                            prof = vehicle_profiles[v_id]
                            prof.update_vehicle_detection(v_type, v_conf, frame_number, v_box)
                            if getattr(prof, "is_stationary_barrier", False):
                                continue
                            match_p = detect_plates_dual_scale(plate_model, frame, v_box, cur_plates, True)
                            p_active += 1
                            if match_p is not None:
                                pw, ph = match_p[2] - match_p[0], match_p[3] - match_p[1]
                                if pw >= config.MIN_PLATE_OCR_WIDTH and ph >= config.MIN_PLATE_OCR_HEIGHT:
                                    p_txt, p_cf, p_sc, p_crp, p_sh = read_plate_text(ocr_reader, frame, match_p, lpr_model)
                                    if p_txt and p_cf >= config.MIN_OCR_CONFIDENCE:
                                        prof.add_reading(p_txt, p_cf, p_sc, p_crp, p_sh, pw, frame_number)
                                        if prof.plate_text and not prof.re_id:
                                            prof.re_id = registry.reid_for_plate(prof.plate_text, current_reid=prof.re_id)
                            draw_vehicle(display_frame, v_box, prof.to_record_dict(), prof.vehicle_type, prof.vehicle_conf)
                            if match_p is not None:
                                draw_plate(display_frame, match_p, prof.plate_text)
                    draw_status(
                        display_frame,
                        f"{video_index + 1}/{video_count} | {video_path.name} (PAUSED -1s)",
                        p_active, len(tracker_to_vehicle_id), displayed_fps, config.PLAYBACK_SPEEDS[playback_speed_index],
                    )
                    cv2.imshow("Vehicle, Plate OCR, and Re-ID", display_frame)
            elif raw_key in (2555904, 83, 65363, ord(".")):  # Right Arrow -> Seek +1s
                target_pos = do_seek(seek_step_frames)
                frame_number = target_pos
                ret, frame = cap.read()
                if ret:
                    display_frame = frame.copy()
                    v_res = vehicle_model.track(
                        frame, persist=True, tracker=config.TRACKER_CONFIG,
                        conf=config.VEHICLE_CONFIDENCE, classes=list(config.VEHICLE_CLASSES),
                        imgsz=config.VEHICLE_IMAGE_SIZE, device=config.DEVICE,
                        agnostic_nms=True, iou=0.45, verbose=False,
                    )
                    p_res = plate_model(frame, conf=config.PLATE_CONFIDENCE, imgsz=config.PLATE_IMAGE_SIZE, device=config.DEVICE, verbose=False)
                    cur_plates = get_plate_boxes(p_res)
                    p_active = 0
                    for result in v_res:
                        for box in result.boxes:
                            if box.id is None:
                                continue
                            c_id = int(box.cls[0])
                            v_box = tuple(map(int, box.xyxy[0]))
                            v_type = refine_vehicle_type(frame, v_box, c_id)
                            v_conf = float(box.conf[0])
                            if not is_valid_vehicle_detection(v_box, v_type, v_conf, frame.shape, frame):
                                continue
                            t_id = int(box.id[0])
                            if t_id not in tracker_to_vehicle_id:
                                tracker_to_vehicle_id[t_id] = registry.new_track_id()
                            v_id = tracker_to_vehicle_id[t_id]
                            if v_id not in vehicle_profiles:
                                vehicle_profiles[v_id] = VehicleTrackerProfile(v_id, video_path.name, frame_number, v_type, v_conf)
                            prof = vehicle_profiles[v_id]
                            prof.update_vehicle_detection(v_type, v_conf, frame_number, v_box)
                            if getattr(prof, "is_stationary_barrier", False):
                                continue
                            match_p = detect_plates_dual_scale(plate_model, frame, v_box, cur_plates, True)
                            p_active += 1
                            if match_p is not None:
                                pw, ph = match_p[2] - match_p[0], match_p[3] - match_p[1]
                                if pw >= config.MIN_PLATE_OCR_WIDTH and ph >= config.MIN_PLATE_OCR_HEIGHT:
                                    p_txt, p_cf, p_sc, p_crp, p_sh = read_plate_text(ocr_reader, frame, match_p, lpr_model)
                                    if p_txt and p_cf >= config.MIN_OCR_CONFIDENCE:
                                        prof.add_reading(p_txt, p_cf, p_sc, p_crp, p_sh, pw, frame_number)
                                        if prof.plate_text and not prof.re_id:
                                            prof.re_id = registry.reid_for_plate(prof.plate_text, current_reid=prof.re_id)
                            draw_vehicle(display_frame, v_box, prof.to_record_dict(), prof.vehicle_type, prof.vehicle_conf)
                            if match_p is not None:
                                draw_plate(display_frame, match_p, prof.plate_text)
                    draw_status(
                        display_frame,
                        f"{video_index + 1}/{video_count} | {video_path.name} (PAUSED +1s)",
                        p_active, len(tracker_to_vehicle_id), displayed_fps, config.PLAYBACK_SPEEDS[playback_speed_index],
                    )
                    cv2.imshow("Vehicle, Plate OCR, and Re-ID", display_frame)
            elif key in (ord("-"), ord("_")):
                playback_speed_index = max(0, playback_speed_index - 1)
            elif key in (ord("+"), ord("=")):
                playback_speed_index = min(len(config.PLAYBACK_SPEEDS) - 1, playback_speed_index + 1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        started_at = time.perf_counter()
        frame_number += 1
        display_frame = frame.copy()

        # Track road vehicles using YOLO + ByteTrack
        vehicle_results = vehicle_model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.VEHICLE_CONFIDENCE,
            classes=list(config.VEHICLE_CLASSES),
            imgsz=config.VEHICLE_IMAGE_SIZE,
            device=config.DEVICE,
            agnostic_nms=True,
            iou=0.45,
            verbose=False,
        )

        plates_are_fresh = frame_number % config.PLATE_DETECTION_INTERVAL == 0
        if plates_are_fresh:
            plate_results = plate_model(
                frame,
                conf=config.PLATE_CONFIDENCE,
                imgsz=config.PLATE_IMAGE_SIZE,
                device=config.DEVICE,
                verbose=False,
            )
            cached_plate_boxes = get_plate_boxes(plate_results)

        active_vehicle_count = 0
        for result in vehicle_results:
            for box in result.boxes:
                if box.id is None:
                    continue

                class_id = int(box.cls[0])
                vehicle_box = tuple(map(int, box.xyxy[0]))
                vehicle_type = refine_vehicle_type(frame, vehicle_box, class_id)
                vehicle_confidence = float(box.conf[0])

                # Reject railings, poles, barricades, and background noise
                if not is_valid_vehicle_detection(vehicle_box, vehicle_type, vehicle_confidence, frame.shape, frame):
                    continue

                tracker_id = int(box.id[0])
                if tracker_id not in tracker_to_vehicle_id:
                    tracker_to_vehicle_id[tracker_id] = registry.new_track_id()

                vehicle_id = tracker_to_vehicle_id[tracker_id]

                if vehicle_id not in vehicle_profiles:
                    vehicle_profiles[vehicle_id] = VehicleTrackerProfile(
                        vehicle_id, video_path.name, frame_number, vehicle_type, vehicle_confidence
                    )
                profile = vehicle_profiles[vehicle_id]
                profile.update_vehicle_detection(vehicle_type, vehicle_confidence, frame_number, vehicle_box)
                if getattr(profile, "is_stationary_barrier", False):
                    continue

                # Dual-scale plate detection (full frame + vehicle bumper zoom)
                matching_plate = detect_plates_dual_scale(
                    plate_model, frame, vehicle_box, cached_plate_boxes, plates_are_fresh
                )

                active_vehicle_count += 1

                if matching_plate is not None:
                    profile.plate_detected = "yes"
                    plate_w = matching_plate[2] - matching_plate[0]
                    plate_h = matching_plate[3] - matching_plate[1]

                    # High-quality OCR triggered when plate is readable
                    # Dynamic frequency: when car is close (plate_w >= FAST_OCR_PLATE_WIDTH), drop retry interval to 3-4 frames!
                    is_approaching = plate_w >= getattr(config, "FAST_OCR_PLATE_WIDTH", 80)
                    retry_interval = config.OCR_FAST_INTERVAL if is_approaching else config.OCR_RETRY_INTERVAL

                    should_run_ocr = (
                        plates_are_fresh
                        and not getattr(profile, "is_locked", False)
                        and plate_w >= config.MIN_PLATE_OCR_WIDTH
                        and plate_h >= config.MIN_PLATE_OCR_HEIGHT
                        and (
                            frame_number >= next_ocr_frame.get(vehicle_id, 0)
                            or plate_w >= profile.best_plate_w * 1.15
                        )
                    )

                    if should_run_ocr:
                        plate_text, ocr_confidence, composite_score, plate_crop, sharpness = read_plate_text(
                            ocr_reader, frame, matching_plate, lpr_model
                        )
                        next_ocr_frame[vehicle_id] = frame_number + retry_interval

                        if plate_text and ocr_confidence >= config.MIN_OCR_CONFIDENCE:
                            prev_plate = profile.plate_text
                            prev_conf = profile.ocr_confidence
                            profile.add_reading(
                                plate_text, ocr_confidence, composite_score, plate_crop, sharpness, plate_w, frame_number
                            )

                            if profile.plate_text:
                                if not profile.re_id:
                                    profile.re_id = registry.reid_for_plate(profile.plate_text, current_reid=profile.re_id)
                                    log_plate_detection(
                                        "DETECTED",
                                        profile.vehicle_id,
                                        profile.re_id,
                                        profile.plate_text,
                                        profile.ocr_confidence,
                                        profile.vehicle_type,
                                        frame_number,
                                        video_path.name,
                                    )
                                    if profile.best_crop is not None:
                                        save_plate_crop(profile.best_crop, profile.re_id)
                                elif profile.plate_text != prev_plate or (profile.ocr_confidence > prev_conf + 0.10):
                                    profile.re_id = registry.reid_for_plate(profile.plate_text, current_reid=profile.re_id)
                                    log_plate_detection(
                                        "REFINED",
                                        profile.vehicle_id,
                                        profile.re_id,
                                        profile.plate_text,
                                        profile.ocr_confidence,
                                        profile.vehicle_type,
                                        frame_number,
                                        video_path.name,
                                        len(profile.frame_readings),
                                    )
                                    if profile.best_crop is not None:
                                        save_plate_crop(profile.best_crop, profile.re_id)

                draw_vehicle(display_frame, vehicle_box, profile.to_record_dict(), profile.vehicle_type, profile.vehicle_conf)
                if matching_plate is not None:
                    draw_plate(display_frame, matching_plate, profile.plate_text)

        frame_seconds = time.perf_counter() - started_at
        current_fps = 1 / frame_seconds if frame_seconds else 0.0
        displayed_fps = current_fps if displayed_fps == 0 else 0.9 * displayed_fps + 0.1 * current_fps
        draw_status(
            display_frame,
            f"{video_index + 1}/{video_count} | {video_path.name}",
            active_vehicle_count,
            len(tracker_to_vehicle_id),
            displayed_fps,
            config.PLAYBACK_SPEEDS[playback_speed_index],
        )
        cv2.imshow("Vehicle, Plate OCR, and Re-ID", display_frame)

        target_frame_seconds = 1 / (source_fps * config.PLAYBACK_SPEEDS[playback_speed_index])
        wait_milliseconds = max(1, int((target_frame_seconds - frame_seconds) * 1000))
        raw_key = cv2.waitKeyEx(wait_milliseconds)
        key = raw_key & 0xFF
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
        if raw_key in (2424832, 81, 65361, ord(",")):  # Left Arrow -> Seek -1s
            frame_number = do_seek(-seek_step_frames)
        if raw_key in (2555904, 83, 65363, ord(".")):  # Right Arrow -> Seek +1s
            frame_number = do_seek(seek_step_frames)
        if key in (ord("-"), ord("_")):
            playback_speed_index = max(0, playback_speed_index - 1)
        if key in (ord("+"), ord("=")):
            playback_speed_index = min(len(config.PLAYBACK_SPEEDS) - 1, playback_speed_index + 1)
        if cv2.getWindowProperty("Vehicle, Plate OCR, and Re-ID", cv2.WND_PROP_VISIBLE) < 1:
            playback_action = "quit"
            break

    cap.release()

    records = []
    plates_read = 0
    for profile in vehicle_profiles.values():
        if getattr(profile, "is_stationary_barrier", False):
            continue
        if profile.best_crop is not None and profile.re_id:
            save_plate_crop(profile.best_crop, profile.re_id)
        rec = profile.to_record_dict()
        records.append(rec)
        if rec["plate_text"]:
            plates_read += 1

    writer.writerows(records)
    print_video_summary(video_path.name, len(vehicle_profiles), plates_read, len(registry.plate_to_reid))

    return playback_action


def main():
    validate_project_files()
    config.configure_torch()
    init_output_dirs()

    video_paths = find_video_paths()
    if not video_paths:
        folders = ", ".join(str(folder) for folder in config.VIDEO_DIRECTORIES)
        raise FileNotFoundError(f"No videos found in: {folders}")

    selected_video_paths = choose_videos(video_paths)
    if not selected_video_paths:
        return

    vehicle_model = YOLO(str(config.VEHICLE_MODEL_PATH))
    plate_model = YOLO(str(config.PLATE_MODEL_PATH))
    ocr_reader = create_ocr_reader()
    parseq_model = create_parseq_model()
    crnn_model = create_lprnet_model()
    lpr_model = parseq_model if parseq_model is not None else crnn_model

    acceleration = "RTX GPU" if config.GPU_AVAILABLE else "CPU fallback"
    print(f"Running on: {acceleration}")
    print(f"Vision Transformer (PARSeq ViT): {'Active on CUDA' if parseq_model is not None else 'Disabled'}")
    crnn_status = f"Active ({config.BASE_DIR / 'best_crnn_bilstm.pth'})" if isinstance(crnn_model, CRNN_BiLSTM) else "Disabled"
    print(f"CRNN-BiLSTM (Indian HSRP): {crnn_status}")
    engine_name = "RapidOCR (PP-OCRv4)" if hasattr(ocr_reader, "text_sys") or type(ocr_reader).__name__ == "RapidOCR" else type(ocr_reader).__name__
    print(f"Primary OCR Engine: {engine_name}")

    registry = IdentityRegistry()
    results_file, writer, results_path = open_csv_writer("unique_vehicles.csv")

    try:
        video_index = 0
        while 0 <= video_index < len(selected_video_paths):
            action = play_video(
                vehicle_model,
                plate_model,
                ocr_reader,
                lpr_model,
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
    finally:
        results_file.close()

    cv2.destroyAllWindows()
    print(f"Saved vehicle and plate results to: {results_path}")


if __name__ == "__main__":
    main()
