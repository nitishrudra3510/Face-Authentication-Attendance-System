import os
import cv2
import numpy as np
import logging
import time

from attendance import punch
from spoof_detection import simple_blink_prompt
from camera_utils import open_camera


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "recognize_face.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.xml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
TARGET_SIZE = (200, 200)

def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """
    Improve robustness under varying lighting using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def load_model():
    """
    Load the trained recognizer and label mapping.
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please run train_model.py first."
        )

    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels mapping not found at {LABELS_PATH}. Please run train_model.py first."
        )

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        logging.error(
            "cv2.face.LBPHFaceRecognizer_create not available. "
            "Ensure opencv-contrib-python is installed."
        )
        raise RuntimeError(
            "cv2.face.LBPHFaceRecognizer_create not available. "
            "Install 'opencv-contrib-python'."
        )

    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
    return recognizer, label_map


def recognize_and_mark(
    confidence_threshold: float = 70.0,
    camera_index: int = 0,
    enable_spoof_detection: bool = True,
) -> None:
    """
    Run real-time face recognition and mark attendance for recognized users.
    """
    recognizer, label_map = load_model()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        logging.error("Haar cascade not found at %s", cascade_path)
        raise FileNotFoundError(f"Haar cascade not found at {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera at index %s", camera_index)
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    logging.info("Starting face recognition loop")
    last_action_ts: dict[str, float] = {}
    per_user_cooldown_seconds = 10.0  # avoid repeated punch attempts every frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = preprocess_gray(gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, TARGET_SIZE)
                label_id, confidence = recognizer.predict(roi_gray)

                name = label_map.get(label_id, "Unknown")

                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0) if confidence < confidence_threshold else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if confidence < confidence_threshold and name != "Unknown":
                    logging.info(
                        "Recognized user %s with confidence %.2f", name, confidence
                    )

                    now = time.time()
                    if name in last_action_ts and (now - last_action_ts[name]) < per_user_cooldown_seconds:
                        # Don't spam punch while the user stays in frame
                        continue

                    if enable_spoof_detection:
                        cv2.putText(
                            frame,
                            "Blink to confirm",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
                        cv2.imshow("Recognition", frame)

                        # Release the current capture before running the blink prompt.
                        # Some camera drivers do not allow multiple simultaneous opens.
                        try:
                            cap.release()
                        except Exception:
                            logging.warning("Failed to release main camera before spoof check")

                        blinked = simple_blink_prompt(camera_index=camera_index)

                        # Re-open the main camera after the blink check
                        cap = open_camera(camera_index)
                        if not cap or not cap.isOpened():
                            logging.error("Failed to reopen camera after spoof check")
                            raise RuntimeError("Failed to reopen camera after spoof check")

                        if not blinked:
                            logging.warning("Spoof suspected for %s (no blink detected)", name)
                            continue

                    action = punch(name)
                    last_action_ts[name] = now
                    cv2.putText(
                        frame,
                        f"{action.replace('_', ' ').upper()} OK",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

            cv2.putText(
                frame,
                "Press q to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Face Recognition - Attendance", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("User ended recognition with 'q'")
                break

    except Exception:
        logging.exception("Unexpected error in recognize_and_mark")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        recognize_and_mark()
    except Exception as e:
        logging.exception("Error in recognize_face main")
        print(f"Error during recognition: {e}")


if __name__ == "__main__":
    main()


