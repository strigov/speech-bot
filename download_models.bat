@echo off
REM Скрипт загрузки моделей для Speech Bot
REM Загружает GigaAM v3 и PyAnnote Speaker Diarization

chcp 65001 >nul 2>&1

echo ============================================
echo   Загрузка моделей Speech Bot
echo ============================================
echo.

REM Проверка виртуального окружения
if not exist "venv" (
    echo ОШИБКА: Виртуальное окружение не найдено!
    echo Сначала запустите setup_environment.bat
    pause
    exit /b 1
)

REM Проверка .env файла
if not exist ".env" (
    echo ВНИМАНИЕ: Файл .env не найден!
    echo.
    echo Для загрузки PyAnnote необходим токен Hugging Face.
    echo 1. Скопируйте .env.example в .env
    echo 2. Добавьте токен HF_TOKEN в .env
    echo.
    echo Или укажите токен через параметр:
    echo   download_models.bat --hf-token hf_ваш_токен
    echo.
)

REM Активация виртуального окружения
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Запуск скрипта загрузки
echo.
echo Запуск загрузки моделей...
echo.

venv\Scripts\python.exe download_models.py %*

REM Сохраняем код возврата
set EXIT_CODE=%ERRORLEVEL%

REM Деактивация
deactivate

echo.
if %EXIT_CODE% EQU 0 (
    echo ============================================
    echo   Загрузка завершена успешно!
    echo ============================================
) else (
    echo ============================================
    echo   Загрузка завершена с ошибками
    echo ============================================
)

echo.
pause
exit /b %EXIT_CODE%
