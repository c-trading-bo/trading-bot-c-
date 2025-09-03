@echo off
REM ========================================
REM LOCAL BOT MECHANIC - INTEGRATED LAUNCHER
REM STRICTLY LOCAL ONLY - NO CLOUD FEATURES
REM Auto-starts with dashboard integration
REM ========================================

setlocal EnableDelayedExpansion

echo ================================================================
echo 🧠 LOCAL BOT MECHANIC - INTEGRATED DASHBOARD LAUNCHER
echo ================================================================
echo STRICTLY LOCAL - runs on YOUR computer only
echo Auto-integrates with main bot dashboard
echo No cloud features, no GitHub minutes used
echo ================================================================

set "BASE_DIR=%~dp0"
set "MECHANIC_DIR=%BASE_DIR%Intelligence\mechanic\local"
set "DASHBOARD_DIR=%BASE_DIR%wwwroot"

REM Check if Python is available
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ ERROR: Python not found in PATH
    echo Please install Python 3.8+ and ensure it's in your PATH
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist "%MECHANIC_DIR%" mkdir "%MECHANIC_DIR%"
if not exist "Intelligence\mechanic\database" mkdir "Intelligence\mechanic\database"

echo.
echo 📍 Starting from: %BASE_DIR%
echo 🔧 Installing/checking dependencies...

REM Install required Python packages
python -m pip install pandas numpy scikit-learn requests yfinance flask --quiet
if !errorlevel! neq 0 (
    echo ⚠️ Some packages may have failed to install, continuing anyway...
)

echo.
echo 🧠 Checking Bot Mechanic files...

REM Check if required files exist
if not exist "%MECHANIC_DIR%\bot_mechanic.py" (
    echo ❌ Missing: bot_mechanic.py
    echo Please ensure the Local Bot Mechanic is properly installed.
    pause
    exit /b 1
)

if not exist "%MECHANIC_DIR%\start_local_mechanic.py" (
    echo ❌ Missing: start_local_mechanic.py
    echo Please ensure the Local Bot Mechanic is properly installed.
    pause
    exit /b 1
)

echo ✅ All required files found

echo.
echo 🚀 Starting Local Bot Mechanic with Dashboard Integration...
echo ================================================================
echo 🌐 Dashboard will be available at: http://localhost:5051
echo 🔗 Will auto-integrate with main dashboard at: http://localhost:5050
echo 🔍 Auto-monitoring and repair enabled
echo ================================================================
echo.

REM Change to base directory
cd /d "%BASE_DIR%"

REM Start the integrated auto-launcher
python "%MECHANIC_DIR%\start_local_mechanic.py"

echo.
echo ================================================================
echo 👋 Local Bot Mechanic stopped
echo ================================================================
pause

if !FILES_OK! equ 0 (
    echo.
    echo ❌ ERROR: Required mechanic files are missing
    echo Please ensure the Local Bot Mechanic installation is complete
    pause
    exit /b 1
)

echo ✅ All required files found
echo.

REM Install required Python packages if not already installed
echo 📦 Checking Python dependencies...
python -c "import flask, threading, json, pathlib" >nul 2>&1
if !errorlevel! neq 0 (
    echo 📦 Installing required Python packages...
    pip install flask requests pandas numpy scikit-learn pathlib
    if !errorlevel! neq 0 (
        echo ❌ Failed to install Python packages
        echo Please install manually: pip install flask requests pandas numpy scikit-learn
        pause
        exit /b 1
    )
    echo ✅ Python packages installed
)

echo ✅ Python dependencies ready
echo.

REM Start the Local Bot Mechanic in background
echo 🧠 Starting Local Bot Mechanic...
start /b "LocalBotMechanic" python "%MECHANIC_DIR%\start_local_mechanic.py" > "%LOG_DIR%\mechanic.log" 2>&1

REM Wait a moment for mechanic to start
timeout /t 5 /nobreak >nul

REM Check if mechanic started successfully
echo 🔍 Checking mechanic status...
python -c "import urllib.request; urllib.request.urlopen('http://localhost:5051/mechanic/health', timeout=5)" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Local Bot Mechanic started successfully
    echo 📊 Dashboard: http://localhost:5051/mechanic/dashboard
    echo 🔗 Health API: http://localhost:5051/mechanic/health
) else (
    echo ⚠️ Mechanic starting up... (may take a moment)
)

echo.
echo ===============================================================
echo 🤖 STARTING MAIN TRADING BOT WITH MECHANIC INTEGRATION
echo ===============================================================
echo.

REM Set environment variables for bot
set "BOT_MECHANIC_ENABLED=true"
set "BOT_MECHANIC_URL=http://localhost:5051"
set "BOT_MODE=%BOT_MODE%"

REM Start the main trading bot
echo 🚀 Starting main trading bot...
echo.

REM Change to the correct directory and start the bot
cd /d "%BASE_DIR%"

REM Check if we have a compiled bot
if exist "src\OrchestratorAgent\bin\Release\net8.0\OrchestratorAgent.exe" (
    echo 🚀 Starting compiled bot...
    "src\OrchestratorAgent\bin\Release\net8.0\OrchestratorAgent.exe"
) else if exist "src\OrchestratorAgent\bin\Debug\net8.0\OrchestratorAgent.exe" (
    echo 🚀 Starting debug bot...
    "src\OrchestratorAgent\bin\Debug\net8.0\OrchestratorAgent.exe"
) else (
    echo 🚀 Building and starting bot...
    dotnet run --project "src\OrchestratorAgent\OrchestratorAgent.csproj"
)

REM If we get here, the bot has stopped
echo.
echo 🛑 Trading bot has stopped
echo.

REM Stop the mechanic when bot stops
echo 🛑 Stopping Local Bot Mechanic...
taskkill /f /im python.exe /fi "WINDOWTITLE eq LocalBotMechanic*" >nul 2>&1

echo.
echo ===============================================================
echo 👋 LOCAL BOT MECHANIC SYSTEM STOPPED
echo ===============================================================
echo.
echo 📝 Logs available in: %LOG_DIR%
echo 📊 Final dashboard: http://localhost:5051/mechanic/dashboard
echo.

pause
