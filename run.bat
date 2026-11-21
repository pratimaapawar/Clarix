@echo off
REM Clarix — start backend and frontend together.
REM Run from the clarix\ folder:  run.bat

echo Starting Clarix backend (port 8000)...
start "Clarix Backend" cmd /k "cd /d %~dp0backend && uvicorn app:app --reload --port 8000"

echo Starting Clarix frontend (port 8501)...
start "Clarix Frontend" cmd /k "cd /d %~dp0frontend && streamlit run app.py"

echo.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:8501
