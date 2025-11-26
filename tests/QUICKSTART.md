# Quick Test Guide

## Самый быстрый способ запустить тест

### 1. Быстрый тест (без моделей, ~1 секунда)

```bash
python run_pipeline_test.py --preprocess-only
```

**Что проверяет:**
- Валидацию и конвертацию аудио
- Работу с файловой системой
- Базовую функциональность пайплайна

**Не требует:**
- HF_TOKEN
- GPU
- Загрузки моделей

---

### 2. Полный тест (с моделями, ~2-5 минут)

```bash
python run_pipeline_test.py
```

**Что проверяет:**
- Весь пайплайн целиком:
  - Preprocessing
  - Diarization (PyAnnote)
  - Transcription (GigaAM)
  - Aggregation
  - Генерация выходных файлов

**Требует:**
- HF_TOKEN в .env файле
- GPU (или будет работать на CPU медленно)
- ~3GB места для моделей (скачиваются один раз)

**Выходные файлы сохраняются в:** `tests/test_result/`

---

## Устранение проблем

### HF_TOKEN не найден

Откройте `.env` файл и добавьте:
```
HF_TOKEN=hf_xxxxxxxxxxxxx
```

Получить токен: https://huggingface.co/settings/tokens

### Нет файла chunk_001.mp3

Убедитесь что файл находится в `tests/chunk_001.mp3`

### CUDA Out of Memory

Измените в `.env`:
```
MODEL_DEVICE=cpu
DIARIZATION_DEVICE=cpu
```

---

## Результаты теста

После успешного прогона в `tests/test_result/` появятся:

- **transcript.txt** - Транскрипция с временными метками и спикерами
- **transcript_plain.txt** - Простой текст
- **transcript.srt** - Субтитры
- **transcript.json** - JSON с метаданными

---

## Альтернативный запуск через pytest

```bash
# Быстрый тест
pytest tests/test_full_pipeline.py::TestFullPipeline::test_pipeline_without_models_validation_only -v -s

# Полный тест
pytest tests/test_full_pipeline.py::TestFullPipeline::test_full_pipeline_with_real_models -v -s
```

---

## Что дальше?

После успешного прохождения тестов можно:
1. Проверить выходные файлы в `tests/test_result/`
2. Запустить основное приложение: `python main.py`
3. Использовать бота в Telegram

Подробная документация: [TESTING.md](../TESTING.md)
