#!/usr/bin/env python3
"""
Скрипт автоматической загрузки моделей для speech-bot.

Загружает:
- GigaAM v3 (распознавание речи)
- PyAnnote Speaker Diarization (диаризация спикеров)

Использование:
    python download_models.py [--hf-token TOKEN] [--device DEVICE]
"""

import argparse
import os
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """Печать заголовка."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, total: int, text: str) -> None:
    """Печать шага."""
    print(f"\n[{step}/{total}] {text}")


def print_success(text: str) -> None:
    """Печать успешного сообщения."""
    print(f"  [OK] {text}")


def print_error(text: str) -> None:
    """Печать ошибки."""
    print(f"  [ОШИБКА] {text}")


def print_warning(text: str) -> None:
    """Печать предупреждения."""
    print(f"  [!] {text}")


def print_info(text: str) -> None:
    """Печать информации."""
    print(f"  {text}")


def get_hf_token() -> str:
    """Получить токен Hugging Face из переменных окружения или .env файла."""
    # Сначала проверяем переменную окружения
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token

    # Пробуем загрузить из .env
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    # Удаляем кавычки если есть
                    token = token.strip('"\'')
                    if token:
                        return token

    return ""


def check_cuda_available() -> tuple[bool, str]:
    """Проверить доступность CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{device_name} ({vram_gb:.1f} GB VRAM)"
        return False, "CUDA недоступен"
    except ImportError:
        return False, "PyTorch не установлен"
    except Exception as e:
        return False, f"Ошибка: {e}"


def download_gigaam(device: str = "cpu") -> bool:
    """Загрузить модель GigaAM v3."""
    print_step(1, 2, "Загрузка GigaAM v3 (распознавание речи)...")
    print_info("Модель: ai-sage/GigaAM-v3")
    print_info("Размер: ~1.5 GB")

    try:
        from transformers import AutoModel
        import torch

        print_info("Загрузка модели...")
        model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            trust_remote_code=True,
        )

        # Проверяем что модель загрузилась корректно
        print_info("Проверка модели...")
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            print_info("Модель загружена на GPU")
        else:
            print_info("Модель загружена на CPU")

        # Освобождаем память
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_success("GigaAM v3 успешно загружен и кэширован!")
        return True

    except Exception as e:
        print_error(f"Не удалось загрузить GigaAM: {e}")
        return False


def download_pyannote(hf_token: str, device: str = "cpu") -> bool:
    """Загрузить модель PyAnnote Speaker Diarization."""
    print_step(2, 2, "Загрузка PyAnnote (диаризация спикеров)...")
    print_info("Модель: pyannote/speaker-diarization-community-1")
    print_info("Размер: ~500 MB")

    if not hf_token:
        print_error("Токен Hugging Face не указан!")
        print_info("")
        print_info("Для загрузки PyAnnote необходим токен Hugging Face.")
        print_info("1. Зарегистрируйтесь на https://huggingface.co/join")
        print_info("2. Создайте токен на https://huggingface.co/settings/tokens")
        print_info("3. Примите лицензию: https://huggingface.co/pyannote/speaker-diarization-community-1")
        print_info("4. Укажите токен в .env файле: HF_TOKEN=hf_ваш_токен")
        print_info("   Или передайте через аргумент: --hf-token hf_ваш_токен")
        return False

    try:
        from pyannote.audio import Pipeline
        import torch

        print_info("Загрузка пайплайна...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )

        # Проверяем что модель загрузилась корректно
        print_info("Проверка модели...")
        if device == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print_info("Модель загружена на GPU")
        else:
            print_info("Модель загружена на CPU")

        # Освобождаем память
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_success("PyAnnote успешно загружен и кэширован!")
        return True

    except Exception as e:
        error_msg = str(e)
        print_error(f"Не удалось загрузить PyAnnote: {e}")

        if "401" in error_msg or "403" in error_msg:
            print_info("")
            print_info("Возможные причины:")
            print_info("- Неверный токен Hugging Face")
            print_info("- Не приняты условия лицензии модели")
            print_info("")
            print_info("Решение:")
            print_info("1. Проверьте токен на https://huggingface.co/settings/tokens")
            print_info("2. Примите лицензию: https://huggingface.co/pyannote/speaker-diarization-community-1")

        return False


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Загрузка моделей для speech-bot"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Токен Hugging Face (или установите HF_TOKEN в .env)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство для проверки моделей (default: auto)"
    )
    parser.add_argument(
        "--skip-gigaam",
        action="store_true",
        help="Пропустить загрузку GigaAM"
    )
    parser.add_argument(
        "--skip-pyannote",
        action="store_true",
        help="Пропустить загрузку PyAnnote"
    )

    args = parser.parse_args()

    print_header("Загрузка моделей для Speech Bot")

    # Определяем токен
    hf_token = args.hf_token or get_hf_token()

    # Определяем устройство
    if args.device == "auto":
        cuda_available, cuda_info = check_cuda_available()
        device = "cuda" if cuda_available else "cpu"
        print_info(f"Устройство: {cuda_info}")
    else:
        device = args.device
        print_info(f"Устройство: {device}")

    # Показываем путь кэша
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print_info(f"Кэш моделей: {cache_dir}")

    results = []

    # Загрузка GigaAM
    if not args.skip_gigaam:
        success = download_gigaam(device)
        results.append(("GigaAM v3", success))

    # Загрузка PyAnnote
    if not args.skip_pyannote:
        success = download_pyannote(hf_token, device)
        results.append(("PyAnnote", success))

    # Итоги
    print_header("Результаты")

    all_success = True
    for name, success in results:
        if success:
            print_success(f"{name}: загружен")
        else:
            print_error(f"{name}: ошибка загрузки")
            all_success = False

    if all_success:
        print_info("")
        print_info("Все модели успешно загружены!")
        print_info("Теперь можно запустить бота: start.bat")
    else:
        print_info("")
        print_warning("Некоторые модели не удалось загрузить.")
        print_info("Проверьте ошибки выше и повторите попытку.")
        sys.exit(1)


if __name__ == "__main__":
    main()
