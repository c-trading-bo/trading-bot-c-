@echo off
REM Clean-view launcher for Windows CMD/double-click
setlocal
powershell -ExecutionPolicy Bypass -NoLogo -NoProfile -File "%~dp0start-clean.ps1" %*
exit /b %ERRORLEVEL%
