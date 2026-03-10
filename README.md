# Бенчмарк LLM для NL2SQL

Минималистичный и воспроизводимый фреймворк для бенчмаркинга LLM на задаче NL2SQL (датасеты формата Spider и BIRD).

Проект построен под notebook-first сценарий:
- запуск экспериментов из Python-модулей в Jupyter
- логика экспериментов хранится в чистых `.py` модулях
- артефакты прогонов сохраняются в JSON для воспроизводимого анализа
- ключевые параметры управляются через переменные окружения (env-first)

## Обзор архитектуры

Конвейер:

`dataset -> prompt -> model -> SQL -> evaluator -> metrics -> results/runs/<timestamp>.json`

Основные принципы:
- минимальные зависимости
- без тяжелых orchestration-фреймворков
- явные интерфейсы и типизированные модули
- воспроизводимый формат артефактов

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── configs/
│   ├── models.yaml
│   └── experiments.yaml
├── datasets/
│   ├── spider/
│   └── bird/
├── src/
│   ├── dataset/
│   │   ├── spider_loader.py
│   │   └── bird_loader.py
│   ├── prompts/
│   │   └── prompt_templates.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── ollama_model.py
│   │   └── api_model.py
│   ├── inference/
│   │   ├── runner.py
│   │   └── batch_runner.py
│   ├── evaluation/
│   │   ├── execution_accuracy.py
│   │   ├── passk.py
│   │   └── sql_executor.py
│   ├── metrics/
│   │   ├── latency.py
│   │   └── token_usage.py
│   ├── logging/
│   │   └── experiment_logger.py
│   └── utils/
│       ├── config_loader.py
│       └── env_loader.py
├── experiments/
│   └── run_experiment.py
├── results/
│   └── runs/
└── notebooks/
    └── analysis.ipynb
```

## Установка

```bash
# запуск из корня проекта
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Опционально editable-установка:

```bash
pip install -e .
```

## Конфигурация через env (env-first)

Все пути и имена моделей можно задать через переменные окружения.

1. Скопируйте шаблон:

```bash
cp .env.example .env
```

2. Отредактируйте значения в `.env`.

3. `.env` загружается автоматически в:
- `experiments/run_experiment.py`
- `src.inference.run_experiment(...)`
- `src.inference.run_batch(...)`

4. `configs/*.yaml` поддерживают интерполяцию:
- `${VAR}` — обязательная переменная окружения
- `${VAR:-default}` — значение по умолчанию

## Формат датасета

Лоадеры ожидают JSONL-записи следующего вида:

```json
{
  "question": "How many users are there?",
  "schema": "Table users(id INTEGER, name TEXT).",
  "gold_sql": "SELECT COUNT(*) FROM users;",
  "db_path": "demo.sqlite"
}
```

Обязательные поля: `question`, `schema`, `gold_sql`.
`db_path` — опционально для каждой записи; также можно задать путь к БД через env/конфиг.

## Конфигурация моделей

`configs/models.yaml` содержит реестр моделей с env-плейсхолдерами:
- `backend: ollama` для локального инференса через `POST /api/generate` (`stream: false`)
- `backend: api` для OpenAI-совместимого `POST /v1/chat/completions`

Для API-бэкендов:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Запуск экспериментов

### Из Python (рекомендуется для Jupyter)

```python
from src.inference.runner import run_experiment
from src.inference.batch_runner import run_batch

result = run_experiment({})
results = run_batch("configs/experiments.yaml")
```

### Из CLI

```bash
python experiments/run_experiment.py --experiment spider_env_demo
python experiments/run_experiment.py --config configs/experiments.yaml
```

Значения по умолчанию для CLI также берутся из env (`L2SB_EXPERIMENTS_CONFIG`, `L2SB_EXPERIMENT`, `L2SB_K`).

## Артефакты результатов

Каждый прогон сохраняется в:

`results/runs/<timestamp>.json`

Минимально обязательные поля:
- `model`
- `dataset`
- `execution_accuracy`
- `pass_at_k`
- `avg_latency`

Также сохраняются:
- `model_name`
- `model_backend`
- `k`, `num_samples`
- агрегированная статистика токенов
- предсказания и флаги оценки по каждой записи

## Метрики оценки

### Точность выполнения (Execution Accuracy)

Для каждого примера:
1. выполняется `gold_sql` в SQLite
2. выполняется предсказанный SQL в той же SQLite БД
3. сравниваются нормализованные результирующие наборы

Итоговая метрика по датасету:
- доля примеров, где результат выполнения предсказания совпадает с gold

### Pass@K (доля успеха в top-k)

Для каждого примера:
- генерируются `k` SQL-кандидатов
- успех фиксируется, если хотя бы один кандидат совпадает с gold по результату выполнения

Итоговая метрика по датасету:
- доля примеров, где найден корректный кандидат в top-k

### Средняя задержка (Average Latency)

- измеряется для каждого вызова генерации (в секундах)
- в отчете используется среднее арифметическое по всем вызовам в прогоне

## Как добавить новую модель

1. Добавьте запись модели в `configs/models.yaml`.
2. При необходимости добавьте адаптер в `src/models/`, реализующий:

```python
class BaseModel:
    def generate(self, prompt: str) -> str:
        ...
```

3. Расширьте фабрику моделей в `src/inference/runner.py`.
4. Добавьте env-переменные для имени модели/URL при необходимости.

## Как добавить новый датасет

1. Реализуйте лоадер в `src/dataset/`, возвращающий нормализованные записи.
2. Зарегистрируйте лоадер в `src/inference/runner.py`.
3. Добавьте JSONL и SQLite файлы в `datasets/<name>/`.
4. Добавьте эксперимент в `configs/experiments.yaml` и нужные env-переменные.

## Работа через Notebook

Используйте `notebooks/analysis.ipynb`, чтобы:
- загрузить `.env`
- запустить `run_experiment`/`run_batch`
- прочитать JSON-результаты через pandas
- построить сводные таблицы и графики для отчета

Автоэкспорт CSV/PNG не включен: единственный источник истины — `results/runs/*.json`.
