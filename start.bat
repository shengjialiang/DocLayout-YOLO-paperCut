@echo off
setlocal

REM DocLayout-YOLO one-click startup (Windows)
REM Usage: double-click, or run .\start.bat from the project root.

REM ---- Config (edit here) ----
set "MODEL_PATH=%~dp0models\doclayout_yolo_docstructbench_imgsz1024.pt"
set "HOST=0.0.0.0"
set "PORT=8000"
set "DEVICE="

REM ---- Optional DocRect (DocTr) model for image enhancement dewarping ----
if not defined DOCRECT_MODEL_PATH (
    set "DOCRECT_MODEL_PATH=%~dp0models\docrect.onnx"
)

cd /d "%~dp0"

REM ---- Pick Python ----
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto :py_ok
)
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    goto :py_ok
)
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found in PATH. Install Python or activate a venv.
    pause
    exit /b 1
)
set "PY=python"
:py_ok

REM ---- Check / install deps ----
%PY% -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing service dependencies: pip install -e ".[service,exam]"
    %PY% -m pip install -e ".[service,exam]"
    if errorlevel 1 (
        echo [ERROR] Dependency install failed. Check network and try manually.
        pause
        exit /b 1
    )
)

REM ---- Check model ----
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found: %MODEL_PATH%
    echo         Download the .pt into models\ or edit MODEL_PATH in this script.
    pause
    exit /b 1
)

echo.
if not exist "%DOCRECT_MODEL_PATH%" (
    echo [INFO] DocRect model not found at %DOCRECT_MODEL_PATH%;
    echo        dewarping stage will be skipped, other stages still active.
)
echo ==========================================
echo  DocLayout-YOLO service starting...
echo  Open:  http://localhost:%PORT%
echo  Press Ctrl+C to stop
echo ==========================================
echo.

set "MODEL_PATH=%MODEL_PATH%"
set "HOST=%HOST%"
set "PORT=%PORT%"
set "DEVICE=%DEVICE%"
set "DOCRECT_MODEL_PATH=%DOCRECT_MODEL_PATH%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

%PY% -m service.app

if errorlevel 1 (
    echo.
    echo [ERROR] Service exited with an error.
)
pause
endlocal
