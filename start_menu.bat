@echo off
title MigrationSafe AI - Controller
color 0B
cd /d "%~dp0"

:menu
cls
echo ================================================================
echo           MIGRATIONSAFE AI - LAUNCH CONTROLLER
echo ================================================================
echo.
echo  [1] Launch Streamlit Web Dashboard (http://localhost:8501)
echo  [2] Open Standalone Offline HTML Report (No Server Needed)
echo  [3] Run 13 Automated Tests (Unit & Integration)
echo  [4] Run 150-Scenario Benchmark Experiment
echo  [5] Exit
echo.
echo ================================================================
set /p choice="Select an option (1-5) [Default is 1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" goto opt1
if "%choice%"=="2" goto opt2
if "%choice%"=="3" goto opt3
if "%choice%"=="4" goto opt4
if "%choice%"=="5" goto opt5

:opt1
cls
echo Starting Streamlit Dashboard...
start "" "http://localhost:8501"
streamlit run app.py
pause
goto menu

:opt2
cls
echo Opening Standalone HTML Report...
start "" "%~dp0MigrationSafeAI_Report.html"
goto menu

:opt3
cls
echo Running 13 Automated Unit and Integration Tests...
echo.
python -m unittest discover -s tests -p "test_*.py" -v
echo.
pause
goto menu

:opt4
cls
echo Running 150-Scenario Measurable Benchmark...
echo.
python experiments/benchmark.py
echo.
pause
goto menu

:opt5
exit
