@echo off
REM Local Bot Mechanic Quick Installer for Windows
REM Automatically sets up and runs your trading bot mechanic

echo.
echo ============================================================
echo              LOCAL BOT MECHANIC INSTALLER
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed or not in PATH
    echo   Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo [+] Python detected
echo [+] Setting up Local Bot Mechanic...

REM Create necessary directories
if not exist "Intelligence\mechanic\database" mkdir "Intelligence\mechanic\database"
if not exist "Intelligence\mechanic\logs" mkdir "Intelligence\mechanic\logs"
if not exist "Intelligence\mechanic\reports" mkdir "Intelligence\mechanic\reports"
if not exist "Intelligence\scripts\strategies" mkdir "Intelligence\scripts\strategies"
if not exist "Intelligence\scripts\ml" mkdir "Intelligence\scripts\ml"
if not exist "Intelligence\scripts\data" mkdir "Intelligence\scripts\data"
if not exist "Intelligence\data" mkdir "Intelligence\data"
if not exist "Intelligence\models" mkdir "Intelligence\models"

echo [+] Directories created

REM Install required packages
echo [+] Installing required Python packages...
pip install pandas numpy scikit-learn requests yfinance ta matplotlib seaborn --quiet

echo [+] Packages installed
echo.

echo Available startup options:
echo 1. Quick Start (Recommended for first run)
echo 2. Full Scan Only
echo 3. Continuous Monitoring Only
echo 4. Dashboard Only
echo 5. Exit

set /p choice="Select option (1-5): "

if "%choice%"=="1" (
    echo.
    echo [*] Starting Quick Start mode...
    python Intelligence\mechanic\quick_start.py
) else if "%choice%"=="2" (
    echo.
    echo [*] Running full scan...
    python Intelligence\mechanic\local\bot_mechanic.py
) else if "%choice%"=="3" (
    echo.
    echo [*] Starting continuous monitoring...
    python -c "from Intelligence.mechanic.local.bot_mechanic import LocalBotMechanic; m=LocalBotMechanic(); m.start_monitoring()"
) else if "%choice%"=="4" (
    echo.
    echo [*] Starting web dashboard...
    python Intelligence\mechanic\dashboard.py
) else (
    echo Goodbye!
    exit /b 0
)

echo.
echo ============================================================
echo              INSTALLATION COMPLETE
echo ============================================================
echo.
echo Your Local Bot Mechanic is now running!
echo.
echo To start again later, run:
echo   python Intelligence\mechanic\quick_start.py
echo.
echo For the web dashboard:
echo   python Intelligence\mechanic\dashboard.py
echo.
pause
