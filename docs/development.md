# aether — Python и uv

## Среды выполнения

Обязательное ограничение: текущая машина — только разработка и ограниченные тесты. Обучение, optimizer steps и переобучение даже одного примера выполняются исключительно в удалённом GPU runtime платного Google Colab. Локальное подключение runtime к Colab не используется. Полные модельные и GPU-проверки также выполняются в Colab; локально используются маленькие forward-фикстуры и mock-компоненты.

Целевой notebook — `notebooks/aether_colab.ipynb`. Он вызывает пакет через `uv run --locked`, а не дублирует тренер в ячейках. Обучающие секции по умолчанию выключены. План подготовки окружения, защиты запуска и постоянного хранения приведён в [tasks.md](tasks.md).

## 1. Обязательный стек

Python 3.12 — начальная минорная версия проекта. Точная patch-версия фиксируется при создании окружения. uv управляет Python, `.venv`, зависимостями и командами. `pyproject.toml` задаёт пакет, `uv.lock` фиксирует разрешённые зависимости; оба файла входят в Git.

Каркас CLI установлен, `--help` и `--version` работают. `train --validate-only` проверяет схему, `inspect --config ... --manifest ...` проверяет синтетический bundle; `infer`, `prepare-data`, `train`, `evaluate` доступны с разрешением удалённого runtime; `serve`/`talk` возвращают `NOT_IMPLEMENTED`; `train` без `--validate-only` возвращает `TRAINING_ENVIRONMENT_REQUIRED`. Семантика команд ниже — целевая, не уже доступная функциональность.

| Слой | Выбор |
|---|---|
| Модель, обучение и тензоры | PyTorch |
| Массивы вне модели | NumPy |
| Веса | safetensors |
| Текстовый токенизатор | SentencePiece, совместимый с весами |
| Проверка конфигураций | Pydantic |
| Сервер и WebSocket | aiohttp |
| Локальный звук | sounddevice; чтение файлов — soundfile |
| CLI | argparse из стандартной библиотеки |
| Тесты | pytest, pytest-asyncio |
| Стиль и статические проверки | Ruff, mypy |

Версии библиотек и источник сборки PyTorch выбираются при проверке целевой платформы и фиксируются lock-файлом. Совместимость CUDA определяется не наличием слова `cuda` в конфигурации, а реальным успешным запуском модели и проверкой драйвера.

## 2. Структура пакета

```text
src/aether/
  __init__.py
  __main__.py
  cli.py
  config.py
  audio/          capture.py, resample.py, buffers.py
  model/          codec.py, quantizer.py, temporal.py, depth.py, embeddings.py
  inference/      engine.py, scheduler.py, state.py, checkpoint.py
  server/         app.py, protocol.py, sessions.py
  client/         local.py
  training/       dataset.py, alignment.py, losses.py, trainer.py
  evaluation/     runner.py, metrics.py, report.py
tests/
  unit/
  integration/
  gpu/
configs/
  model/
  runtime/
  training/
docs/
notebooks/
  aether_colab.ipynb
```

Модель не импортирует сервер, CLI или аудиоустройства. Сервер не реализует семплирование токенов. Подготовка данных не зависит от живых сессий. Общие контракты определяются один раз.

## 3. Окружение

После создания конфигурации проекта:

```bash
uv python install 3.12
uv python pin 3.12
uv sync --locked --group dev
uv run --locked aether --help
```

`uv sync --locked` проверяет актуальность lock-файла. Обычный `uv sync` может обновить его. Обновления зависимостей выполняются отдельным изменением с просмотром diff. В CI применяются `--locked` и та же зафиксированная версия uv.

Добавление зависимости выполняется через `uv add`, инструментов разработки — через `uv add --group dev`. Не использовать ручной `pip install` внутри проектного окружения как способ изменять состав проекта. Системные аудиобиблиотеки и GPU-драйверы устанавливаются отдельно и фиксируются в инструкции окружения.

## 4. Группы зависимостей

- Основной пакет: вывод модели, конфигурации, загрузка весов и сервер.
- `dev`: тестирование, форматирование и проверка типов.
- `model`: закреплённый Python backend, PyTorch/torchaudio 2.8 и зависимости весов. Устанавливается только в Colab.
- `train`: включает `model`; локальные тесты не устанавливают эту группу.
- `audio`: локальное устройство ввода-вывода и файлы.

Это dependency groups в `pyproject.toml`. Примеры ниже предполагают, что они определены. Не добавлять все экспериментальные библиотеки в основную группу.

```bash
uv sync --locked --group dev --group audio
uv run --locked --group dev ruff check .
uv run --locked --group dev ruff format --check .
uv run --locked --group dev mypy src/aether
uv run --locked --group dev pytest
```

## 5. Целевой CLI

```bash
uv run --locked aether inspect --config configs/model/base.json
uv run --locked aether infer --config configs/model/base.json --input sample.wav --output answer.wav
uv run --locked aether serve --config configs/runtime/local.json
uv run --locked --group audio aether talk --url ws://127.0.0.1:8998/v1/session
uv run --locked aether evaluate --config configs/evaluation.json
```

Эти модельные команды с полными весами выполняются в Colab; локально используются тестовые конфигурации. Только внутри удалённого Colab runtime, после проверки ресурсов и явного включения обучения:

```bash
uv run --locked --group train aether train --config configs/training/adapter.json
```

Локально разрешена только валидация обучающей конфигурации через `train --validate-only`, без создания optimizer и запуска обучения.

`inspect` проверяет файлы, размеры, словари, dtype, устройство и память, не открывая микрофон. `infer` сохраняет также текст и манифест. `serve` прогревает worker до готовности. `talk` явно открывает микрофон. `evaluate` не изменяет веса. `train` пишет в новую директорию запуска, не перезаписывая предыдущий эксперимент.

## 6. Стиль реализации

Публичные функции получают аннотации типов. В документации тензорных API указываются shape, dtype, устройство, диапазон значений и владелец состояния. Конфигурации валидируются до загрузки крупных весов. Неизвестные ключи считаются ошибкой.

Контекст `torch.inference_mode()` применяется для вывода, модель переводится в `eval()`. Глобальное изменяемое состояние сессии запрещено. Audio callback не выполняет inference, файловую запись и сетевое ожидание; он только передаёт данные через ограниченную очередь.

Оптимизации компиляцией и CUDA graphs добавляются после корректного eager-вывода. Для каждой оптимизации сохраняется возможность сравнения с простой реализацией.

## 7. Воспроизводимость

Манифест запуска содержит commit, состояние изменённых файлов, хэш `uv.lock`, версии Python и uv, версию PyTorch, GPU, драйвер, dtype, конфигурацию, seed и хэши всех весов и токенизатора. Одинаковый seed не гарантирует побитового совпадения на разных устройствах; сравнение имеет заданные допуски.

В Git не включаются `.venv`, веса, кеши, записи микрофона, локальные секреты и большие отчёты. Маленькие синтетические фикстуры и схемы данных включаются. Названия проекта в пользовательском интерфейсе, пакете и документах — `aether`.
