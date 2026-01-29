@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Face Authentication Attendance System - Windows Runner
REM - Avoids PowerShell ExecutionPolicy issues by using CMD
REM - Creates venv if missing, installs requirements, runs scripts
REM ==========================================================

cd /d "%~dp0"

echo [1/4] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found in PATH. Install Python and check "Add Python to PATH".
  echo Then reopen terminal and run this again.
  exit /b 1
)

echo [2/4] Creating venv (if needed)...
if not exist "venv\Scripts\python.exe" (
  python -m venv venv
  if errorlevel 1 (
    echo Failed to create venv.
    exit /b 1
  )
)

echo [3/4] Installing dependencies...
"venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo Failed to upgrade pip.
  exit /b 1
)
"venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 (
  echo Failed to install requirements.
  exit /b 1
)

echo [4/4] Running project steps...
echo.
echo --- Step 1: Register face ---
"venv\Scripts\python.exe" register_face.py
if errorlevel 1 (
  echo register_face.py failed. Check logs\register_face.log
  exit /b 1
)

echo.
echo --- Step 2: Train model ---
"venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
  echo train_model.py failed. Check logs\train_model.log
  exit /b 1
)

echo.
echo --- Step 3: Recognize + mark attendance ---
"venv\Scripts\python.exe" recognize_face.py
if errorlevel 1 (
  echo recognize_face.py failed. Check logs\recognize_face.log
  exit /b 1
)

echo.
echo DONE.
echo - Attendance file: attendance.csv
echo - Logs folder: logs\
endlocal


