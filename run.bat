@echo off
title MigrationSafe AI - Windows Launcher
color 0A
cd /d "%~dp0"

echo ================================================================
echo           MIGRATIONSAFE AI - START WINDOW APPLICATION
echo ================================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not detected in your system PATH!
    echo Please install Python 3.10+ from python.org and check "Add to PATH".
    echo.
    pause
    exit /b 1
)

echo [OK] Python is available.
echo.

:: 2. Check Streamlit dependency
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Dependency installation failed!
        pause
        exit /b 1
    )
)

echo [OK] Libraries verified.
echo.
echo ================================================================
echo Opening Web Browser at http://localhost:8501 ...
echo Starting Streamlit Local Server...
echo ================================================================
echo.

start "" "http://localhost:8501"
python -m streamlit run app.py --server.port 8501

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application exited.
)
pause
