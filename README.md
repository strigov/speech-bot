# Telegram-бот для транскрибации аудио

Telegram-бот для высококачественной транскрибации аудио с использованием **GigaAM v3** (распознавание речи) и **PyAnnote** (диаризация спикеров). Оптимизирован для Windows и видеокарт NVIDIA с поддержкой CUDA.

## Возможности

- **Высокоточное распознавание речи**: GigaAM v3 — современная модель для русской речи
- **Диаризация спикеров**: PyAnnote определяет и разделяет разных говорящих
- **Длинные аудиозаписи**: Обработка файлов до нескольких часов с интеллектуальным разбиением
- **Поддержка GPU**: Ускорение на видеокартах NVIDIA (CUDA 12.x)
- **Система очередей**: Управление параллельными запросами с приоритизацией
- **Админ-команды**: Мониторинг GPU и состояния очереди

## Системные требования

- **ОС**: Windows 10 или 11 (x64)
- **GPU**: NVIDIA с 12+ ГБ VRAM (рекомендуется RTX 3080/4070/5070 или выше)
- **Драйвер**: NVIDIA Driver 535+ (для RTX 50xx — версия 581.80+)
- **Накопитель**: SSD с минимум 50 ГБ свободного места
- **Софт**:
  - Python 3.10.x (строгое требование)
  - FFmpeg (должен быть в PATH)
  - Git (опционально)

## Быстрая установка

### 1. Клонируйте репозиторий

```batch
git clone <url-репозитория>
cd speech-bot
```

### 2. Запустите установку окружения

```batch
setup_environment.bat
```

Скрипт автоматически:
- Создаст виртуальное окружение Python
- Установит PyTorch с поддержкой CUDA
- Установит все зависимости
- Создаст необходимые директории

### 3. Загрузите модели ML

```batch
download_models.bat
```

**Важно**: Перед запуском скрипта вам потребуется:

1. **Токен Hugging Face** — получите на https://huggingface.co/settings/tokens
2. **Принять лицензию PyAnnote** — перейдите по ссылке https://huggingface.co/pyannote/speaker-diarization-community-1 и нажмите "Agree"

### 4. Настройте конфигурацию

Отредактируйте файл `.env`:

```ini
# Обязательные настройки
TELEGRAM_BOT_TOKEN=ваш_токен_бота
HF_TOKEN=ваш_токен_huggingface
```

### 5. Запустите бота

```batch
start.bat
```

---

## Подробная инструкция по установке моделей

### Используемые модели

| Модель | Назначение | Размер | Источник |
|--------|-----------|--------|----------|
| GigaAM v3 | Распознавание речи (ASR) | ~1.5 ГБ | [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) |
| PyAnnote Speaker Diarization | Определение спикеров | ~500 МБ | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |

### Автоматическая загрузка

Запустите скрипт загрузки:

```batch
download_models.bat
```

Или Python-версию напрямую:

```batch
venv\Scripts\activate.bat
python download_models.py
```

### Ручная загрузка (если автоматическая не сработала)

#### GigaAM v3

GigaAM загружается автоматически при первом запуске через `transformers`:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ai-sage/GigaAM-v3", trust_remote_code=True)
```

Модель кэшируется в `~/.cache/huggingface/hub/`.

#### PyAnnote Speaker Diarization

1. **Зарегистрируйтесь на Hugging Face**: https://huggingface.co/join

2. **Создайте токен доступа**:
   - Перейдите на https://huggingface.co/settings/tokens
   - Нажмите "New token"
   - Выберите тип "Read"
   - Скопируйте токен

3. **Примите лицензионное соглашение**:
   - Откройте https://huggingface.co/pyannote/speaker-diarization-community-1
   - Нажмите кнопку "Agree and access repository"

4. **Укажите токен в `.env`**:
   ```ini
   HF_TOKEN=hf_ваш_токен_здесь
   ```

5. Модель загрузится автоматически при первом запуске

### Расположение кэша моделей

По умолчанию модели кэшируются в:

- **Windows**: `C:\Users\<пользователь>\.cache\huggingface\hub\`
- **Размер кэша**: ~3-5 ГБ после загрузки всех моделей

Для изменения директории кэша установите переменную окружения:

```batch
set HF_HOME=D:\models\huggingface
```

---

## Команды бота

### Пользовательские команды

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие и проверка статуса |
| `/help` | Справка по использованию |
| `/status` | Текущая позиция в очереди |
| `/cancel` | Отмена текущей обработки |

### Админ-команды

| Команда | Описание |
|---------|----------|
| `/admin gpu` | Использование GPU-памяти |
| `/admin queue` | Статистика очереди |
| `/admin mode` | Переключение публичный/приватный режим |

---

## Конфигурация (.env)

### Telegram

```ini
# Токен бота от @BotFather
TELEGRAM_BOT_TOKEN=your_token_here

# Опционально: для файлов >20 МБ (до 2 ГБ)
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_USE_CLIENT_API=true
```

### Модели и GPU

```ini
# Токен Hugging Face
HF_TOKEN=hf_your_token_here

# Устройство: cuda или cpu
MODEL_DEVICE=cuda
DIARIZATION_DEVICE=cuda

# Размер батча (уменьшите при OOM)
ASR_BATCH_SIZE=32
DIARIZATION_BATCH_SIZE=32
```

### Ограничения

```ini
# Максимальная длительность аудио (минуты)
MAX_AUDIO_DURATION_MINUTES=180

# Максимальный размер файла (МБ)
MAX_FILE_SIZE_MB=500

# Лимит VRAM (ГБ) — оставьте запас для системы
MAX_VRAM_GB=14.0
```

### Контроль доступа

```ini
# ID администраторов (через запятую)
TELEGRAM_ADMIN_USER_IDS=123456789,987654321

# Разрешённые чаты (пусто = все)
TELEGRAM_ALLOWED_CHAT_IDS=123456789,-1001234567890

# Режим доступа: true = бот доступен всем, false = только разрешённым чатам
BOT_PUBLIC_MODE=false
```

**Режимы работы бота:**
- **Приватный** (`BOT_PUBLIC_MODE=false`, по умолчанию) — бот обрабатывает аудио только от пользователей и в чатах, перечисленных в `TELEGRAM_ALLOWED_CHAT_IDS`
- **Публичный** (`BOT_PUBLIC_MODE=true`) — бот доступен всем пользователям без ограничений

Режим можно переключать на лету через админ-панель (`/admin` → кнопка режима или `/admin mode`), без перезапуска бота.

**Как узнать ID:**
- Личный ID: напишите боту @userinfobot
- ID группы: добавьте @RawDataBot в группу

---

## Поддержка больших файлов (>20 МБ)

Стандартный Telegram Bot API ограничивает размер загружаемых файлов до **20 МБ**. Для обработки файлов до **2 ГБ** необходимо использовать Telegram Client API (MTProto).

### Шаг 1: Получение API credentials

1. Откройте https://my.telegram.org и войдите по номеру телефона

2. Введите код подтверждения, который придёт в Telegram

3. Нажмите **"API development tools"**

4. Если приложение ещё не создано, заполните форму:
   - **App title**: любое название (например, "Speech Bot")
   - **Short name**: короткое имя латиницей (например, "speechbot")
   - **Platform**: можно выбрать "Desktop"
   - **Description**: опционально

5. Нажмите **"Create application"**

6. Скопируйте полученные данные:
   - **App api_id** — числовой ID (например, `12345678`)
   - **App api_hash** — строка из 32 символов (например, `a1b2c3d4e5f6...`)

### Шаг 2: Настройка .env

Добавьте полученные данные в файл `.env`:

```ini
# Telegram Client API для больших файлов
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
TELEGRAM_USE_CLIENT_API=true
```

### Шаг 3: Первый запуск

При первом запуске с Client API бот попросит авторизацию:

1. Введите номер телефона аккаунта, от имени которого будет работать бот
2. Введите код подтверждения из Telegram
3. Если включена двухфакторная аутентификация — введите пароль

Сессия сохраняется, повторная авторизация не потребуется.

### Шаг 4: Настройка Privacy Mode (для групп)

Если бот будет работать в группах, отключите Privacy Mode:

1. Напишите @BotFather в Telegram
2. Отправьте `/setprivacy`
3. Выберите вашего бота
4. Выберите **Disable**

Это позволит боту видеть все сообщения в группе, а не только команды.

### Ограничения и особенности

| Параметр | Bot API | Client API |
|----------|---------|------------|
| Макс. размер файла | 20 МБ | 2 ГБ |
| Требует авторизации | Нет | Да (один раз) |
| Скорость загрузки | Средняя | Высокая |

**Важно:**
- API credentials привязаны к вашему аккаунту Telegram
- Не передавайте `api_id` и `api_hash` третьим лицам
- При подозрении на утечку — отзовите ключи на my.telegram.org

---

## Решение проблем

### "CUDA not available"

1. Проверьте установку драйвера:
   ```batch
   nvidia-smi
   ```

2. Проверьте версию PyTorch:
   ```batch
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

3. Переустановите PyTorch с CUDA:
   ```batch
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### "Out of Memory" (OOM)

1. Уменьшите размер батча в `.env`:
   ```ini
   ASR_BATCH_SIZE=16
   DIARIZATION_BATCH_SIZE=16
   ```

2. Уменьшите лимит VRAM:
   ```ini
   MAX_VRAM_GB=12.0
   ```

3. Переключитесь на CPU (медленнее, но стабильно):
   ```ini
   MODEL_DEVICE=cpu
   DIARIZATION_DEVICE=cpu
   ```

### "FFmpeg not found"

1. Скачайте FFmpeg: https://ffmpeg.org/download.html
2. Распакуйте в удобную папку
3. Добавьте путь к `bin` в переменную PATH
4. Или укажите путь в `.env`:
   ```ini
   FFMPEG_PATH=C:/path/to/ffmpeg/bin
   ```

### RTX 50xx (Blackwell) — особенности

Видеокарты RTX 5070/5080/5090 требуют PyTorch nightly:

```batch
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

При проблемах используйте CPU-режим до выхода стабильной версии PyTorch.

### Модели не загружаются

1. Проверьте интернет-соединение
2. Убедитесь что токен HF_TOKEN верный
3. Проверьте что приняли лицензию PyAnnote
4. Попробуйте очистить кэш:
   ```batch
   rmdir /s /q "%USERPROFILE%\.cache\huggingface"
   ```

---

## Структура проекта

```
speech-bot/
├── main.py                 # Точка входа
├── start.bat               # Запуск бота
├── setup_environment.bat   # Установка окружения
├── download_models.bat     # Загрузка моделей
├── download_models.py      # Скрипт загрузки моделей
├── .env                    # Конфигурация (не в git)
├── .env.example            # Пример конфигурации
├── requirements.txt        # Python-зависимости
├── config/                 # YAML-конфигурации
├── src/
│   ├── bot/                # Telegram-обработчики
│   ├── pipeline/           # ML-пайплайн
│   │   ├── transcriber.py  # GigaAM ASR
│   │   ├── diarizer.py     # PyAnnote диаризация
│   │   ├── preprocessor.py # Предобработка аудио
│   │   └── aggregator.py   # Сборка результатов
│   └── utils/              # Утилиты
├── temp/                   # Временные файлы
└── logs/                   # Логи
```

---

## Лицензия

MIT License
