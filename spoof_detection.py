import os
import cv2
import time
import math
import logging
from typing import Tuple, Optional

from camera_utils import open_camera


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "spoof_detection.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(eye_pts: list[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio (EAR) using 6 landmarks:
      p1..p6 around the eye
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    if len(eye_pts) != 6:
        return 1.0
    p1, p2, p3, p4, p5, p6 = eye_pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


def _try_mediapipe_facemesh():
    """
    Import mediapipe lazily. Returns (mp, FaceMesh) or (None, None).
    """
    try:
        import mediapipe as mp  # type: ignore

        return mp, mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # enables iris/eye refinement
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        return None, None


def liveness_blink_mediapipe(
    timeout: float = 6.0,
    camera_index: int = 0,
    ear_threshold: float = 0.19,
    consec_frames: int = 2,
) -> bool:
    """
    Real blink detection using MediaPipe FaceMesh (EAR-based).
    Returns True if a blink is detected within timeout.
    """
    mp, face_mesh = _try_mediapipe_facemesh()
    if face_mesh is None:
        logging.info("MediaPipe not available; cannot run EAR blink detection")
        return False

    # FaceMesh landmark indices around eyes (commonly used subset)
    # Left eye: [33, 160, 158, 133, 153, 144]
    # Right eye: [362, 385, 387, 263, 373, 380]
    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]

    cap = open_camera(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera for mediapipe liveness at index %s", camera_index)
        return False

    start_time = time.time()
    counter = 0
    blinked = False

    try:
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            cv2.putText(
                frame,
                "Blink to verify liveness (MediaPipe)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                left_eye = [(lm[i].x * w, lm[i].y * h) for i in left_idx]
                right_eye = [(lm[i].x * w, lm[i].y * h) for i in right_idx]

                ear_val = (_ear(left_eye) + _ear(right_eye)) / 2.0

                cv2.putText(
                    frame,
                    f"EAR: {ear_val:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                if ear_val < ear_threshold:
                    counter += 1
                else:
                    if counter >= consec_frames:
                        blinked = True
                        break
                    counter = 0

            cv2.imshow("Spoof Detection - Blink", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception:
        logging.exception("Unexpected error in liveness_blink_mediapipe")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            face_mesh.close()
        except Exception:
            pass

    logging.info("Mediapipe blink result=%s", blinked)
    return blinked


def simple_blink_prompt(timeout: float = 5.0, camera_index: int = 0) -> bool:
    """
    Very lightweight spoof prevention:
    - Ask the user to blink within `timeout` seconds.
    - Detects large changes in eye region intensity as a proxy.
    - Returns True if blink-like change is detected, False otherwise.
    """
    # Prefer real blink detection if MediaPipe is installed; otherwise fallback to motion heuristic
    blinked_mp = liveness_blink_mediapipe(timeout=timeout, camera_index=camera_index)
    if blinked_mp:
        return True

    cap = open_camera(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera for spoof detection at index %s", camera_index)
        return False

    logging.info("Starting simple blink prompt for spoof detection")
    start_time = time.time()

    prev_gray = None
    blink_detected = False

    try:
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame during spoof detection")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute absolute difference as crude motion detector
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = diff.mean()

                cv2.putText(
                    frame,
                    "Blink to verify liveness",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Spoof Detection - Blink", frame)

                # Motion threshold; tune as needed
                if motion_score > 10:  # arbitrary heuristic
                    blink_detected = True
                    logging.info("Blink/motion detected, score=%.2f", motion_score)
                    break

            else:
                cv2.imshow("Spoof Detection - Blink", frame)

            prev_gray = gray

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("User cancelled spoof detection with 'q'")
                break

    except Exception:
        logging.exception("Unexpected error in simple_blink_prompt")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return blink_detected


def main():
    print("Look at the camera and blink when prompted.")
    result = simple_blink_prompt()
    if result:
        print("Liveness confirmed (blink detected).")
    else:
        print("Liveness not confirmed (no blink detected).")


if __name__ == "__main__":
    main()


