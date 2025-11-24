@echo off
REM First-time setup script for GigaAM & PyAnnote Telegram Transcriber

echo ============================================
echo  Environment Setup
echo  GigaAM ^& PyAnnote Telegram Transcriber
echo ============================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10.x from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

REM Check if version starts with 3.10
echo %PYTHON_VERSION% | findstr /b "3.10" >nul
if errorlevel 1 (
    echo WARNING: Python 3.10.x is recommended for best compatibility
    echo Current version: %PYTHON_VERSION%
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: FFmpeg is not installed or not in PATH
    echo Please install FFmpeg and add it to your PATH
    echo Download from: https://ffmpeg.org/download.html
    echo.
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA 12.1 support...
echo This may take a while...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install -r requirements.txt

REM Create .env from example if not exists
if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo Creating .env from .env.example...
        copy .env.example .env
        echo.
        echo IMPORTANT: Please edit .env and add your:
        echo   - TELEGRAM_BOT_TOKEN
        echo   - HF_TOKEN ^(Hugging Face token for PyAnnote^)
    )
)

REM Create directories
echo.
echo Creating directories...
if not exist "temp\audio" mkdir temp\audio
if not exist "temp\results" mkdir temp\results
if not exist "temp\checkpoints" mkdir temp\checkpoints
if not exist "logs" mkdir logs

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env and add your TELEGRAM_BOT_TOKEN and HF_TOKEN
echo 2. Run start.bat to start the bot
echo 3. Or run start_dev.bat for development mode
echo.

deactivate
pause
