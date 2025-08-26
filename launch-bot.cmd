@echo off
REM Simple wrapper to run the existing PowerShell launcher from CMD
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0launch-bot.ps1" %*
exit /b %ERRORLEVEL%