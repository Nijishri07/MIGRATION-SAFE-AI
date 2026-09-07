@echo off
title MigrationSafe AI Dashboard
color 0A
cd /d "%~dp0"
echo ================================================================
echo           STARTING MIGRATIONSAFE AI DASHBOARD...
echo ================================================================
echo.
echo 1. Opening Web Browser at http://localhost:8501 ...
start "" "http://localhost:8501"
echo 2. Launching Streamlit Server...
echo.
streamlit run app.py
pause
