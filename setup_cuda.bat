@echo off
REM CUDA installation helper for GigaAM & PyAnnote Telegram Transcriber

echo ============================================
echo  CUDA Setup Helper
echo ============================================
echo.

REM Check NVIDIA driver
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: NVIDIA driver not found
    echo Please install the latest NVIDIA driver from:
    echo https://www.nvidia.com/Download/index.aspx
    echo.
    pause
    exit /b 1
)

echo NVIDIA Driver found:
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
echo.

REM Check CUDA toolkit
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA Toolkit not found in PATH
    echo.
    echo The bot can still work with PyTorch's bundled CUDA libraries,
    echo but for optimal performance, install CUDA Toolkit 12.1:
    echo https://developer.nvidia.com/cuda-12-1-0-download-archive
    echo.
) else (
    echo CUDA Toolkit found:
    nvcc --version | findstr "release"
    echo.
)

REM Test PyTorch CUDA
echo Testing PyTorch CUDA support...
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    deactivate
) else (
    echo Virtual environment not found. Run setup_environment.bat first.
)

echo.
echo ============================================
echo  Recommended Configuration
echo ============================================
echo.
echo For RTX 5070 Ti (16GB VRAM):
echo   MAX_VRAM_GB=14.0
echo   ASR_BATCH_SIZE=32
echo   DIARIZATION_BATCH_SIZE=32
echo.
echo These values leave 2GB for system overhead.
echo.

pause
