"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we lock onto the
     likely active face within each camera shot and move a bounded crop window.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_FFMPEG_PATH, LOCAL_OUTPUT_DIR


_YUNET_MODEL_PATH = Path(__file__).with_name("models") / "face_detection_yunet_2023mar.onnx"


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _pick_locked_face(
    faces: List[Tuple[int, int, int, int]],
    locked_x: Optional[float],
    reacquire: bool,
    max_distance: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Acquire the largest face, then keep the face nearest the current lock."""
    if not faces:
        return None
    if reacquire or locked_x is None:
        return max(faces, key=lambda face: face[2] * face[3])

    nearest = min(faces, key=lambda face: abs((face[0] + face[2] / 2) - locked_x))
    if abs((nearest[0] + nearest[2] / 2) - locked_x) <= max_distance:
        return nearest
    return None


def _pick_active_face(
    faces: List[Tuple[int, int, int, int]],
    activities: List[float],
    locked_x: Optional[float],
    reacquire: bool,
    max_distance: float,
    min_activity: float = 0.012,
    switch_margin: float = 0.004,
) -> Optional[Tuple[int, int, int, int]]:
    """Prefer a persistently moving mouth without abandoning a valid lock."""
    if not faces:
        return None

    active_index = max(range(len(faces)), key=lambda index: activities[index])
    if reacquire or locked_x is None:
        if activities[active_index] >= min_activity:
            return faces[active_index]
        return max(faces, key=lambda face: face[2] * face[3])

    current_index = min(
        range(len(faces)),
        key=lambda index: abs((faces[index][0] + faces[index][2] / 2) - locked_x),
    )
    current = faces[current_index]
    current_distance = abs((current[0] + current[2] / 2) - locked_x)
    if current_distance > max_distance:
        return faces[active_index] if activities[active_index] >= min_activity else max(
            faces, key=lambda face: face[2] * face[3]
        )

    if (
        active_index != current_index
        and activities[active_index] >= min_activity
        and activities[active_index] >= activities[current_index] + switch_margin
    ):
        return faces[active_index]
    return current


def _face_patch(cv2, gray, face, top: float, bottom: float):
    """Return a fixed-size center-face patch for motion comparison."""
    x, y, w, h = face
    x0, x1 = max(0, int(x + w * 0.18)), min(gray.shape[1], int(x + w * 0.82))
    y0, y1 = max(0, int(y + h * top)), min(gray.shape[0], int(y + h * bottom))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(gray[y0:y1, x0:x1], (32, 20), interpolation=cv2.INTER_AREA)


def _update_face_activities(cv2, gray, faces, tracks: List[Dict]) -> List[float]:
    """Track faces and estimate speech from mouth motion minus head motion."""
    activities: List[float] = []
    used_tracks = set()

    for face in faces:
        x, y, w, h = face
        center = (x + w / 2, y + h / 2)
        candidates = []
        for index, track in enumerate(tracks):
            if index in used_tracks or track["missed"] > 2:
                continue
            tx, ty = track["center"]
            distance = ((center[0] - tx) ** 2 + (center[1] - ty) ** 2) ** 0.5
            candidates.append((distance, index))

        match_index = None
        if candidates:
            distance, index = min(candidates)
            if distance <= max(w, h) * 0.8:
                match_index = index

        mouth = _face_patch(cv2, gray, face, 0.52, 0.92)
        upper = _face_patch(cv2, gray, face, 0.12, 0.50)
        activity = 0.0
        if match_index is not None:
            track = tracks[match_index]
            if (
                mouth is not None
                and upper is not None
                and track["mouth"] is not None
                and track["upper"] is not None
            ):
                mouth_motion = cv2.mean(cv2.absdiff(mouth, track["mouth"]))[0] / 255.0
                upper_motion = cv2.mean(cv2.absdiff(upper, track["upper"]))[0] / 255.0
                raw_activity = max(0.0, mouth_motion - upper_motion * 0.55)
                activity = track["activity"] * 0.55 + raw_activity * 0.45
            track.update(
                center=center,
                mouth=mouth,
                upper=upper,
                activity=activity,
                missed=0,
            )
            used_tracks.add(match_index)
        else:
            tracks.append({
                "center": center,
                "mouth": mouth,
                "upper": upper,
                "activity": activity,
                "missed": 0,
            })
            used_tracks.add(len(tracks) - 1)
        activities.append(activity)

    for index, track in enumerate(tracks):
        if index not in used_tracks:
            track["missed"] += 1
            track["activity"] *= 0.55
    tracks[:] = [track for track in tracks if track["missed"] <= 3]
    return activities


def _create_face_detector(cv2, input_size: Tuple[int, int]):
    """Prefer YuNet's CNN detector; retain Haar as a no-model fallback."""
    if _YUNET_MODEL_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            detector = cv2.FaceDetectorYN.create(
                str(_YUNET_MODEL_PATH),
                "",
                input_size,
                0.7,
                0.3,
                5000,
            )
            return "yunet", detector
        except Exception as error:
            print(f"[clip/local] YuNet unavailable, using Haar fallback: {error}", flush=True)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return "haar", detector


def _detect_faces(detector, frame, gray, min_face_size: int) -> List[Tuple[int, int, int, int]]:
    kind, model = detector
    if kind == "yunet":
        model.setInputSize((frame.shape[1], frame.shape[0]))
        _, detected = model.detect(frame)
        if detected is None:
            return []
        faces = [tuple(int(value) for value in face[:4]) for face in detected]
    else:
        detected = model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        faces = [tuple(int(value) for value in face) for face in detected]

    return [face for face in faces if face[2] >= min_face_size and face[3] >= min_face_size]


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        LOCAL_FFMPEG_PATH, "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_detector = _create_face_detector(cv2, (src_w, src_h))
    min_face_size = max(40, round(min(src_w, src_h) * 0.06))

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    detect_every = max(1, round(fps / 10))
    reacquire_after = max(2, round(0.75 * fps / detect_every))
    max_face_distance = max(crop_w * 0.45, src_w * 0.12)
    scene_cut_threshold = 32.0
    dead_zone = crop_w * 0.08
    max_pan_per_frame = max(2.0, src_w * 0.012)

    frame_index = 0
    previous_scene_frame = None
    locked_x: Optional[float] = None
    pan_x = src_w / 2.0
    missed_detections = reacquire_after
    face_tracks: List[Dict] = []
    speaker_candidate_x: Optional[float] = None
    speaker_candidate_hits = 0
    speaker_switch_hits = 3
    fast_pan_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_frame = cv2.resize(gray, (64, 36))
        scene_cut = (
            previous_scene_frame is not None
            and cv2.mean(cv2.absdiff(previous_scene_frame, scene_frame))[0] >= scene_cut_threshold
        )
        previous_scene_frame = scene_frame

        if scene_cut:
            locked_x = None
            pan_x = src_w / 2.0
            missed_detections = reacquire_after
            face_tracks.clear()
            speaker_candidate_x = None
            speaker_candidate_hits = 0

        if scene_cut or frame_index % detect_every == 0:
            faces = _detect_faces(face_detector, frame, gray, min_face_size)
            activities = _update_face_activities(cv2, gray, faces, face_tracks)
            face = _pick_active_face(
                faces,
                activities,
                locked_x,
                reacquire=locked_x is None or missed_detections >= reacquire_after,
                max_distance=max_face_distance,
            )
            if face is None:
                missed_detections += 1
            else:
                x, _, w, _ = face
                face_x = x + w / 2.0
                current_visible = locked_x is None or any(
                    abs((candidate[0] + candidate[2] / 2) - locked_x) <= max_face_distance
                    for candidate in faces
                )
                is_new_speaker = locked_x is not None and abs(face_x - locked_x) > max_face_distance

                if is_new_speaker and current_visible:
                    if (
                        speaker_candidate_x is not None
                        and abs(face_x - speaker_candidate_x) <= max_face_distance
                    ):
                        speaker_candidate_hits += 1
                    else:
                        speaker_candidate_x = face_x
                        speaker_candidate_hits = 1
                    if speaker_candidate_hits >= speaker_switch_hits:
                        locked_x = face_x
                        fast_pan_frames = round(fps * 0.8)
                        speaker_candidate_x = None
                        speaker_candidate_hits = 0
                else:
                    locked_x = face_x if locked_x is None else locked_x + (face_x - locked_x) * 0.25
                    speaker_candidate_x = None
                    speaker_candidate_hits = 0

                if scene_cut or not current_visible:
                    locked_x = face_x
                    pan_x = face_x
                missed_detections = 0

        if locked_x is not None:
            delta = locked_x - pan_x
            if abs(delta) > dead_zone:
                pan_limit = max_pan_per_frame * (2.0 if fast_pan_frames > 0 else 1.0)
                step = max(-pan_limit, min(pan_limit, delta * 0.12))
                pan_x += step
        fast_pan_frames = max(0, fast_pan_frames - 1)

        x0 = max(0, min(src_w - crop_w, int(pan_x - crop_w / 2)))
        y0 = max(0, (src_h - crop_h) // 2)
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)
        frame_index += 1

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        LOCAL_FFMPEG_PATH, "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
