@echo off
REM Development launcher with hot reload for GigaAM & PyAnnote Telegram Transcriber

echo ============================================
echo  GigaAM ^& PyAnnote Telegram Transcriber
echo  DEVELOPMENT MODE
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if .env exists, create from example if not
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env
        echo Please configure your .env file with your tokens.
        pause
    ) else (
        echo ERROR: Neither .env nor .env.example found
        pause
        exit /b 1
    )
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

REM Add FFmpeg to PATH for this session
set "PATH=%PATH%;C:\Program Files\DownloadHelper CoApp"

REM Set development environment variables
set DEBUG=true
set HOT_RELOAD=true
set LOG_LEVEL=DEBUG
set CUDA_VISIBLE_DEVICES=0

REM Run with auto-reload using watchdog (if installed)
echo.
echo Starting bot in development mode...
echo Press Ctrl+C to stop
echo.

REM Try to use watchmedo for auto-reload, fall back to regular python
where watchmedo >nul 2>&1
if %errorlevel% equ 0 (
    watchmedo auto-restart --directory=. --pattern="*.py" --recursive -- python main.py
) else (
    echo Note: Install watchdog for auto-reload: pip install watchdog
    python main.py
)

deactivate
