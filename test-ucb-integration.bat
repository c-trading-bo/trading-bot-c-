@echo off
REM 🧪 SIMULATION MODE TEST - Test integrated UCB without TopstepX
REM Tests Python UCB service auto-launch + C# integration

echo.
echo ╔═══════════════════════════════════════════════════════════════════════════════════════╗
echo ║                        🧪 SIMULATION MODE UCB INTEGRATION TEST 🧪                     ║
echo ║                                                                                       ║
echo ║  🎯 Tests integrated Python UCB service launch without TopstepX connection           ║
echo ║  🐍 Verifies C# ↔ Python communication pipeline                                     ║
echo ║  🔬 Validates dual UCB system (C# Neural UCB + Python FastAPI)                      ║
echo ╚═══════════════════════════════════════════════════════════════════════════════════════╝
echo.

REM Disable TopstepX connection for simulation
set TOPSTEPX_JWT=
set TOPSTEPX_USERNAME=
set TOPSTEPX_API_KEY=

REM Enable UCB integration
set ENABLE_UCB=1
set UCB_PORT=5000
set UCB_SERVICE_URL=http://localhost:5000
set SIMULATION_MODE=1

echo 🎯 Simulation Mode Configuration:
echo    • TopstepX credentials: DISABLED (for testing)
echo    • UCB integration: ENABLED
echo    • UCB port: %UCB_PORT%
echo    • Mode: SIMULATION ONLY
echo.

echo 🚀 Starting integrated UCB test...
echo 📊 Watch for Python UCB service auto-launch logs
echo.

cd src\UnifiedOrchestrator
dotnet run

echo.
echo 🧪 UCB Integration test complete
pause
