## Face Authentication Attendance System (Assignment Notes)

### Model and approach used
- **Face detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Face recognition**: OpenCV **LBPH** (`cv2.face.LBPHFaceRecognizer_create`)
- **Preprocessing for lighting**: CLAHE (contrast-limited adaptive histogram equalization) applied to grayscale frames
- **Spoof prevention (basic)**: Motion/blink-like check using frame-difference within a short time window

This is a practical baseline system: fast, offline, easy to run locally. It is **not** state-of-the-art and is not secure against advanced spoofing.

---

### Training process
1. Run `register_face.py` to capture ~50 grayscale face crops into `dataset/<user_id>/`.
2. Run `train_model.py`:
   - Loads all images from `dataset/*/`
   - Resizes to a consistent `200x200`
   - Trains LBPH
   - Saves model to `models/face_recognizer.xml`
   - Saves labels mapping to `models/labels.npy`

---

### Attendance process (Punch-In / Punch-Out)
- During recognition, if the face is confidently recognized and liveness check passes:
  - First successful action of the day → **Punch-In**
  - Second successful action of the day → **Punch-Out**
  - Further attempts that day → no change (already done)

CSV format (v2): `name,date,punch_in,punch_out`

Backward compatibility:
- If an old `attendance.csv` exists in v1 format (`name,datetime`), punch logs are written to `attendance_v2.csv`.

---

### Accuracy expectations
LBPH is sensitive to:
- Lighting changes
- Pose/angle changes
- Occlusion (mask/hand)
- Camera quality

Typical expectation for a small demo (few users, controlled capture):
- **Good** recognition when users are registered in similar conditions to runtime.
- **Degraded** performance in low light / strong backlight / side profile.

---

### Known failure cases / limitations (ML + system)
- **Spoofing**: The current liveness check is basic; photos/videos may still bypass it.
- **Look-alikes**: Similar faces can be confused.
- **Haar Cascade**: Can miss faces in extreme angles or low light.
- **Single-camera assumption**: Uses one webcam index; may need index change on some PCs.

---

### Practical reliability notes
- If the camera doesn’t open, close other camera apps and try a different camera index.
- If `cv2.face` is missing, ensure `opencv-contrib-python` is installed (not `opencv-python`).

