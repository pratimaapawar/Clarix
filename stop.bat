@echo off
REM Clarix — free any ports left held by old runs (backend 8000, web app 8502).
REM Run this anytime you get "address already in use" / WinError 10048.

powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000,8502 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
echo Freed ports 8000 and 8502.
