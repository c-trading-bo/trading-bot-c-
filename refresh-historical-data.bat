@echo off
REM Refresh historical data in incremental mode
REM This fetches only new bars since last update

cd /d "%~dp0"

echo ========================================
echo Historical Data Refresh - Incremental Mode
echo ========================================
echo.

REM Set incremental mode
set REFRESH_MODE=incremental

REM Run Python script
python fetch-and-save-historical-data.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✅ Historical data refresh completed successfully
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ❌ Historical data refresh failed with error code %ERRORLEVEL%
    echo ========================================
)

REM Keep window open if run manually (close if run by Task Scheduler)
if "%1"=="" pause
