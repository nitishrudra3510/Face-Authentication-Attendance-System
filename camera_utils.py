import os
import sys
import cv2


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Open a webcam in a cross-platform way.

    - On Windows, prefer DirectShow.
    - On macOS, try AVFoundation first then fall back.
    - On other OSes, use the default backend.

    Returns an opened `cv2.VideoCapture` (may still be closed; caller must check `isOpened()`).
    """
    # Windows: use DirectShow backend which is often more reliable
    if os.name == "nt":
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            return cap
        except Exception:
            pass

    # macOS: try AVFoundation, then default
    if sys.platform == "darwin":
        # AVFoundation is the recommended macOS backend for many OpenCV builds
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            if cap and cap.isOpened():
                return cap
            # if not opened, release and continue
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass

    # Generic fallback: let OpenCV pick default backend
    try:
        cap = cv2.VideoCapture(camera_index)
        return cap
    except Exception:
        # As last resort, return a VideoCapture object (may be closed)
        return cv2.VideoCapture(camera_index)


