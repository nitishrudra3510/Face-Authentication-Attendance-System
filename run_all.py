import os
import sys
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_DIR))


def main() -> int:
    """
    Run the full project flow using the *current* Python interpreter:
      1) register_face.py
      2) train_model.py
      3) recognize_face.py

    Usage examples:
      - Interactive register: python run_all.py
      - Non-interactive: python run_all.py --name Alice --samples 20
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run full face attendance flow")
    parser.add_argument("--name", help="Name to register (if provided, will run register_face.py)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to capture when registering")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--skip-register", action="store_true", help="Skip the register step")
    args = parser.parse_args()

    print("Face Authentication Attendance System - Runner")
    print(f"Python: {sys.executable}")
    print(f"Project: {PROJECT_DIR}")

    try:
        if not args.skip_register:
            if args.name:
                run([sys.executable, "register_face.py", args.name, "--samples", str(args.samples), "--camera", str(args.camera)])
            else:
                # If name not provided, run register interactively (will prompt inside script)
                run([sys.executable, "register_face.py"])

        run([sys.executable, "train_model.py"])
        run([sys.executable, "recognize_face.py"])
        print("\nDONE.")
        print("- Attendance: attendance.csv")
        print("- Logs: logs/")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED with exit code {e.returncode}")
        print("Check logs/ for details.")
        return e.returncode
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


