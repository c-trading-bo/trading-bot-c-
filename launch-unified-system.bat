@echo off
REM 🚀 UNIFIED TRADING SYSTEM LAUNCHER 🚀
REM One command to launch everything - C# UnifiedOrchestrator + Python UCB service!

echo.
echo ╔═══════════════════════════════════════════════════════════════════════════════════════╗
echo ║                          🚀 UNIFIED TRADING SYSTEM LAUNCHER 🚀                        ║
echo ║                                                                                       ║
echo ║  🧠 ONE COMMAND - Launches UnifiedOrchestrator + Python UCB service together         ║
echo ║  ⚡ INTEGRATED - No more manual Python service startup                                ║
echo ║  🔄 COORDINATED - Services start in correct order with health checks                 ║
echo ║  🌐 AUTOMATIC - UCB service launches as background service                           ║
echo ╚═══════════════════════════════════════════════════════════════════════════════════════╝
echo.

echo 🔧 Checking workspace structure...
if not exist "src\UnifiedOrchestrator\UnifiedOrchestrator.csproj" (
    echo ❌ Error: Not in correct workspace directory
    echo Please run this from the root workspace directory containing src\UnifiedOrchestrator\
    pause
    exit /b 1
)

if not exist "python\ucb\ucb_api.py" (
    echo ❌ Error: Python UCB service not found at python\ucb\ucb_api.py
    echo Please ensure the UCB Python service is properly installed
    pause
    exit /b 1
)

echo ✅ Workspace structure verified

echo.
echo 🎯 Environment Configuration:
echo    • ENABLE_UCB: %ENABLE_UCB% (default: enabled)
echo    • UCB_PORT: %UCB_PORT% (default: 5000)
echo    • UCB_SERVICE_URL: %UCB_SERVICE_URL% (default: http://localhost:5000)
echo    • TOPSTEPX_JWT: %TOPSTEPX_JWT:~0,20%... (TopstepX authentication)

REM Set default environment variables if not set
if "%ENABLE_UCB%"=="" set ENABLE_UCB=1
if "%UCB_PORT%"=="" set UCB_PORT=5000
if "%UCB_SERVICE_URL%"=="" set UCB_SERVICE_URL=http://localhost:5000

echo.
echo 🚀 Launching Unified Trading System...
echo.
echo 📝 What will happen:
echo    1. UnifiedOrchestrator starts up
echo    2. PythonUcbLauncher detects UCB service needed
echo    3. Python UCB FastAPI service auto-launches at localhost:%UCB_PORT%
echo    4. UCBManager connects to Python service via HTTP
echo    5. Dual UCB system active (C# Neural UCB + Python UCB service)
echo    6. All systems coordinated and ready for trading!
echo.

echo ⏳ Starting in 3 seconds... (Ctrl+C to cancel)
timeout /t 3 /nobreak >nul

echo.
echo 🎉 LAUNCHING UNIFIED TRADING ORCHESTRATOR...
echo 📊 Monitor logs for Python UCB service auto-startup
echo 🔍 Health check: http://localhost:%UCB_PORT%/health
echo.

REM Launch the UnifiedOrchestrator - Python UCB service will auto-start
cd src\UnifiedOrchestrator
dotnet run

echo.
echo 🛑 Unified Trading System shutdown complete
pause
