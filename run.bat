@echo off
REM ── GridLock Brain: one-click run ──────────────────────────────
REM 1) regenerate intelligence from the raw CSV  2) open the dashboard
cd /d "%~dp0"
echo Running GridLock Brain pipeline...
python pipeline.py
if %errorlevel% neq 0 (
  echo Pipeline failed. Ensure Python + pandas/numpy are installed: pip install -r requirements.txt
  pause
  exit /b 1
)
echo Running validation suite...
python validate.py
echo Opening dashboard...
start "" "dashboard\index.html"
