## **Face Authentication Attendance System**

Camera-based face authentication with **attendance CSV logging** and a **liveness (blink/motion) check**.

### **Files**
- **`register_face.py`**: capture face samples and save to `dataset/<user_id>/`
- **`train_model.py`**: train LBPH model and save to `models/`
- **`recognize_face.py`**: recognize faces and update `attendance.csv`
- **`attendance.py`**: attendance CSV with **Punch-In / Punch-Out** (per day)
- **`spoof_detection.py`**: lightweight liveness check
- **`camera_utils.py`**: cross-platform camera opening helper
- **`run_all.cmd`**: one-click runner for Windows (avoids PowerShell ExecutionPolicy issues)

---

## **Run (Windows - fastest)**

Open **PowerShell** in the project folder and run:

```powershell
.\run_all.cmd
```

If you’re using **CMD**, run:

```cmd
run_all.cmd
```

This will:
- Create `venv` (if missing)
- Install requirements
- Run: `register_face.py` → `train_model.py` → `recognize_face.py`

---

## **(Optional) Migrate old attendance.csv**

If you have an old `attendance.csv` with `name,datetime` and you want to convert it into Punch-In/Out format:

```powershell
venv\Scripts\python.exe migrate_attendance.py
```

It will create/update `attendance_v2.csv`.

---

## **Run (no venv activation, PowerShell-safe)**

If PowerShell blocks `Activate.ps1`, do **not** activate venv. Run Python directly:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
venv\Scripts\python.exe run_all.py
```

---

## **Run (manual)**

### **1) Create venv**

```powershell
python -m venv venv
```

### **2) Install dependencies (no activation needed)**

```powershell
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### **3) Step-by-step**

```powershell
venv\Scripts\python.exe register_face.py
venv\Scripts\python.exe train_model.py
venv\Scripts\python.exe recognize_face.py
```

---

## **Expected Output**

- **Registration**: camera window opens, saves ~50 images to `dataset/<user_id>/`
- **Training**: creates `models/face_recognizer.xml` and `models/labels.npy`
- **Recognition**: camera window opens, recognized name appears, **Punch-In / Punch-Out** saved to `attendance.csv` (or `attendance_v2.csv` if upgrading from old format)

---

## **Common Errors**

### **PowerShell says: running scripts is disabled**
- Don’t activate venv in PowerShell.
- Use `.\run_all.cmd` **or** run with `venv\Scripts\python.exe ...` commands as shown above.

### **Camera not opening**
- Close apps using camera (Teams/Zoom/browser).
- Try changing camera index in code (0 → 1 → 2).

### **`AttributeError: cv2 has no attribute face`**
- Install contrib build:
  - `pip install opencv-contrib-python`

### **MediaPipe install issues**
- MediaPipe is optional but improves blink detection.
- If install fails, the project will still run (it falls back to motion-based liveness).

---

## **Logs**

Logs are saved in the `logs/` folder (one log per script).

---

## **Assignment Deliverables**

- **Working demo**: Run locally using `.\run_all.cmd` (Windows) or run scripts individually.
- **Complete codebase**: Included in this folder.
- **Documentation**:
  - See `docs.md` for:
    - model + approach
    - training process
    - accuracy expectations
    - known failure cases
# Face-Authentication-Attendance-System
