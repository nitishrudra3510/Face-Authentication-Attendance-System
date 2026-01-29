
# Face Authentication Attendance System

## Overview
This project is a Python-based face authentication attendance system that uses your webcam to recognize faces and log attendance automatically. It is designed for small teams, classrooms, or demo use, and runs fully offline on your computer.

**Key Features:**
- Register new users by capturing face images from webcam
- Train a face recognition model (LBPH, OpenCV)
- Real-time face recognition and attendance logging
- Basic spoof prevention (blink/motion check)
- Attendance stored in CSV (with punch-in/punch-out)
- Cross-platform (Windows, macOS, Linux)

## How it Works
1. **Register Face:** Capture face images for each user using your webcam. Images are saved in `dataset/<name>/`.
2. **Train Model:** Train a face recognizer on the collected images. Model and label mapping are saved in `models/`.
3. **Recognize & Mark Attendance:** The system recognizes faces in real-time, checks for liveness (blink), and logs attendance in `attendance.csv`.

## Quick Start
1. Install requirements:
	```bash
	python3 -m pip install -r requirements.txt
	```
2. Register a user (replace `YourName`):
	```bash
	python3 register_face.py YourName --samples 20
	```
3. Train the model:
	```bash
	python3 train_model.py
	```
4. Start recognition and attendance:
	```bash
	python3 recognize_face.py
	```

Or run the full flow (non-interactive):
```bash
python3 run_all.py --name YourName --samples 20
```

## Attendance CSV Format
- `attendance.csv` columns: `name,date,punch_in,punch_out`
- Each user can punch in and out once per day.

## Requirements
- Python 3.8+
- Webcam
- Packages: see `requirements.txt`

## Security & Limitations
- Liveness check is basic (blink/motion); not secure against advanced spoofing.
- Recognition accuracy depends on lighting, camera quality, and registration conditions.

## Project Structure
- `register_face.py` — Capture face images for a user
- `train_model.py` — Train the face recognizer
- `recognize_face.py` — Real-time recognition and attendance
- `attendance.py` — Attendance CSV logic
- `spoof_detection.py` — Liveness (blink) detection
- `camera_utils.py` — Cross-platform camera open helper
- `models/` — Trained model and label mapping
- `dataset/` — Collected face images
- `logs/` — Log files for debugging

## Credits
- Uses OpenCV, NumPy, and optionally MediaPipe for blink detection

---
**For more details, see `docs.md`.**
