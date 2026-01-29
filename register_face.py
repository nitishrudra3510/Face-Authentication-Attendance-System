import os
import sys
import cv2
from pathlib import Path
from typing import Optional

from camera_utils import open_camera


DATASET_DIR = Path("dataset")
TARGET_SIZE = (200, 200)


def ensure_dataset_dir(name: str) -> Path:
    """Create (or reuse) a directory under `dataset/` for the given user name."""
    DATASET_DIR.mkdir(exist_ok=True)
    person_dir = DATASET_DIR / name
    person_dir.mkdir(exist_ok=True)
    return person_dir


def next_image_index(person_dir: Path) -> int:
    existing = [p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except Exception:
            continue
    return max(nums) + 1 if nums else 1


def capture_faces(name: str, samples: int = 20, camera_index: int = 0) -> None:
    """Open camera, detect faces and save `samples` face images for `name`."""
    person_dir = ensure_dataset_dir(name)
    start_idx = next_image_index(person_dir)

    cap = open_camera(camera_index)
    if not cap or not cap.isOpened():
        raise RuntimeError("Could not open the camera. Check your device and permissions.")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        cap.release()
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    print(f"Capturing {samples} samples for '{name}'. Press 'q' to quit early.")
    count = 0
    idx = start_idx

    try:
        while count < samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Retrying...")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            # Draw rectangles for user feedback and save the largest face found
            if len(faces) > 0:
                # choose largest face by area
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_img = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_img, TARGET_SIZE)

                img_path = person_dir / f"{name}_{idx:03d}.jpg"
                cv2.imwrite(str(img_path), face_resized)
                count += 1
                idx += 1
                print(f"Saved {count}/{samples}: {img_path}")

            cv2.putText(frame, f"Saved: {count}/{samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Register Face - Press q to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Interrupted by user.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (KeyboardInterrupt).")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Finished. Collected {count} images for '{name}' in: {person_dir}")


def main(argv: Optional[list] = None) -> int:
    argv = argv or sys.argv[1:]

    import argparse

    parser = argparse.ArgumentParser(description="Register a person's face samples into dataset/")
    parser.add_argument("name", help="Person name (will be used as folder name under dataset/)")
    parser.add_argument("--samples", type=int, default=20, help="Number of face samples to capture (default: 20)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    args = parser.parse_args(argv)

    try:
        capture_faces(args.name, samples=args.samples, camera_index=args.camera)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
