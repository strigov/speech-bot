@echo off
REM Production launcher for GigaAM & PyAnnote Telegram Transcriber

echo ============================================
echo  GigaAM ^& PyAnnote Telegram Transcriber
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10.x
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Check if .env exists
if not exist ".env" (
    echo ERROR: .env file not found
    echo Please copy .env.example to .env and configure your settings
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup_environment.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Set environment variables for CUDA
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Run the application
echo.
echo Starting bot...
echo.
python main.py

REM Deactivate on exit
deactivate
