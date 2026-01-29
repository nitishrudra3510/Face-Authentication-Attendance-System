import os
import cv2
import numpy as np
import logging
from typing import Tuple, List


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train_model.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.xml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
TARGET_SIZE = (200, 200)


def load_images_and_labels(dataset_dir: str = DATASET_DIR) -> Tuple[List[np.ndarray], List[int], dict]:
    """
    Load grayscale face images and integer labels from the dataset directory.
    Assumes each subfolder is a user_id, containing jpg images of faces.
    Returns images, labels, and label-to-name mapping dict.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist")

    faces: List[np.ndarray] = []
    labels: List[int] = []
    label_map: dict = {}  # int -> name

    current_label = 0

    for person_name in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name
        logging.info("Processing user '%s' with label %d", person_name, current_label)

        for file_name in os.listdir(person_path):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning("Failed to read image %s", img_path)
                continue

            img = cv2.resize(img, TARGET_SIZE)

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if not faces:
        raise RuntimeError("No faces found in dataset. Please register at least one user.")

    return faces, labels, label_map


def train_and_save_model():
    """
    Train an LBPH face recognizer and save the trained model and labels mapping.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        faces, labels, label_map = load_images_and_labels()
    except Exception:
        logging.exception("Failed to load images and labels")
        raise

    logging.info("Loaded %d face images for training", len(faces))

    # Create LBPH face recognizer (works across platforms if OpenCV is built with contrib)
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

    faces_np = [np.asarray(face, dtype=np.uint8) for face in faces]
    labels_np = np.asarray(labels, dtype=np.int32)

    recognizer.train(faces_np, labels_np)
    recognizer.save(MODEL_PATH)
    np.save(LABELS_PATH, label_map)

    logging.info("Model trained and saved to %s", MODEL_PATH)
    logging.info("Labels mapping saved to %s", LABELS_PATH)


def main():
    try:
        train_and_save_model()
        print("Model training completed successfully.")
        print(f"Model saved at: {MODEL_PATH}")
        print(f"Labels mapping saved at: {LABELS_PATH}")
    except Exception as e:
        logging.exception("Error in train_model main")
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()


