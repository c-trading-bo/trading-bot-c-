@echo off
REM Production UCB API Startup Script for Windows
REM Run this to start the Neural UCB FastAPI service

echo 🚀 Starting UCB FastAPI Service...
echo 📁 Working directory: %CD%

REM Check if virtual environment exists
if not exist "venv\" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Set environment variables for production
set UCB_HOST=0.0.0.0
set UCB_PORT=5000
set UCB_PERSISTENCE_PATH=ucb_state.pkl
set UCB_WEIGHTS_PATH=neural_ucb_topstep.pth

echo 🌐 Starting UCB service on %UCB_HOST%:%UCB_PORT%
echo 💾 Persistence file: %UCB_PERSISTENCE_PATH%
echo 🧠 Model weights: %UCB_WEIGHTS_PATH%
echo.
echo ⚠️  IMPORTANT NOTES:
echo    - Keep this SINGLE-PROCESS (no --workers ^> 1)
echo    - UCB stats live in memory and sync across requests
echo    - Run locally (no VPS/VPN) for TopstepX compliance
echo.

REM Start the service
python ucb_api.py

pause
