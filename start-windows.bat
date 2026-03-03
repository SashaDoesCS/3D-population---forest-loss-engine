@echo off
echo ========================================
echo   Global Forest Analyzer
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed!
    echo Please download and install Node.js from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

REM Check if npm modules are installed
if not exist "node_modules" (
    echo [Setup] Installing dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [Info] Starting application...
echo.

REM Start the Electron app
npm start

pause
