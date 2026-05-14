# Экспериментальный стенд для NL2SQL и генерации кода

Репозиторий содержит экспериментальный стенд для сравнения двух классов
языковых моделей в задачах генерации SQL-запросов и Python-кода.

- `M1` - крупные general-purpose LLM, доступные через API.
- `M2` - компактные специализированные модели, запускаемые локально через
  Ollama.

Стенд объединяет подготовку данных, запуск инференса, вычисление метрик,
сохранение raw-ответов и построение отчетных таблиц/графиков. Основной
исследовательский домен - `nl2sql/`; домен `code/` использует ту же общую
инфраструктуру для независимой задачи генерации программного кода.

## Структура репозитория

```text
.
├── nl2sql/                     # Text-to-SQL: Spider, BIRD, SEB
├── code/                       # Генерация Python-кода: HumanEval+, MBPP+
├── shared/                     # Общие конфиги, backends, logging, статистика
├── data/                       # Локальные данные benchmark-ов
├── results/                    # Raw-ответы, метрики, графики, архивы
├── reports/                    # Материалы для приложений и подписей
├── final_nl2sql_analysis.ipynb # Итоговый аналитический ноутбук NL2SQL
├── pyproject.toml              # Зависимости и настройки проекта
└── uv.lock                     # Lock-файл окружения
```

Общие компоненты находятся в `shared/`:

- `shared/configs/models.yaml` - единый каталог моделей и их параметров;
- `shared/inference/` - транспорт к API, Anthropic и Ollama;
- `shared/evaluation/` - статистические вспомогательные функции;
- `shared/config.py`, `shared/logging_utils.py` - загрузка YAML и вывод
  прогресса.

## Домены

| Домен | Назначение | Benchmark-и | Основные режимы |
| --- | --- | --- | --- |
| `nl2sql/` | Генерация SQL по вопросу на естественном языке и схеме БД | `Spider`, `BIRD`, `SEB` | `ea`, `pass_k` |
| `code/` | Генерация Python-функций по условию задачи | `HumanEval+`, `MBPP+` | `fc`, `pass_k` |

Подробные команды и формат артефактов описаны в доменных README:

- [`nl2sql/README.md`](nl2sql/README.md)
- [`code/README.md`](code/README.md)

## Модели

Модели задаются в [`shared/configs/models.yaml`](shared/configs/models.yaml).
Ключи модели используются в CLI-аргументе `--model`.

### Основной NL2SQL-набор

Этот набор используется для сопоставления результатов `ea` и `pass_k`.

| Класс | Ключ | Отображаемое имя | Backend |
| --- | --- | --- | --- |
| `M1` | `m1_deepseek` | `DeepSeek V3.2` | API |
| `M1` | `m1_chatgpt` | `ChatGPT 5.2` | API |
| `M2` | `m2_arctic` | `Arctic-Text2SQL-R1-7B` | Ollama |
| `M2` | `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | Ollama |
| `M2` | `m2_hrida` | `Hrida-T2SQL q8_0` | Ollama |

`m1_claude` (`Claude Sonnet 4.5`) хранится в общем каталоге моделей как
дополнительная API-модель. В NL2SQL-артефактах она представлена в
EA-результатах и не входит в основной набор `pass_k`.

### Code-domain модели

Для домена генерации кода используются модели с `supports_code: true`:
`m1_deepseek`, `m1_chatgpt`, `m1_claude`, `m2_qwen2_5_coder`,
`m2_qwen2_5_coder_14b`, `m2_deepseek_coder` и дополнительные модели,
отмеченные в конфиге как неактивные по умолчанию.

## Окружение

Проект рассчитан на Python 3.11+ и управление зависимостями через `uv`.

```bash
uv sync
cp .env.example .env
```

В `.env` указываются ключи API-провайдеров. Для локальных моделей нужен
запущенный Ollama и заранее загруженные модели, указанные в
`shared/configs/models.yaml`.

Базовая проверка окружения:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Воспроизведение NL2SQL-результатов

Подготовка данных:

```bash
uv run python nl2sql/scripts/01_download_data.py --benchmark all
```

Запуск инференса для основного NL2SQL-набора:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode ea
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode ea

uv run python nl2sql/scripts/02_run_inference.py --model m1_deepseek --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m1_chatgpt --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_arctic --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark all --mode pass_k
uv run python nl2sql/scripts/02_run_inference.py --model m2_hrida --benchmark all --mode pass_k
```

Пересчет метрик по существующим raw-файлам:

```bash
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label ea
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label pass_k
```

Быстрая проверка пайплайна:

```bash
uv run python nl2sql/scripts/02_run_inference.py --model m2_defog --benchmark spider --mode ea --limit 10
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label ea
```

## Воспроизведение code-domain результатов

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
uv run python code/scripts/03_evaluate.py --run-label fc
uv run python code/scripts/03_evaluate.py --run-label pass_k
```

Smoke-run для проверки окружения:

```bash
uv run python code/scripts/02_run_inference.py --model m1_deepseek --benchmark humaneval_plus --mode fc --limit 5 --mini
```

## Артефакты

| Путь | Содержимое |
| --- | --- |
| `results/nl2sql/raw/` | Raw JSONL-файлы NL2SQL-инференса |
| `results/nl2sql/metrics/ea/` | Execution Accuracy, sample-level метрики и сводки |
| `results/nl2sql/metrics/pass_k/` | Pass@K, sample-level метрики и сводки |
| `results/nl2sql/figures/` | Графики для NL2SQL-отчетов |
| `results/nl2sql/synthetic_benchmark/` | Результаты synthetic e-commerce benchmark |
| `results/code/raw/` | Raw JSONL-файлы code-domain инференса |
| `results/code/metrics/` | Метрики Functional Correctness и Pass@K |
| `results/code/figures/` | Графики code-domain |
| `results/*/archive/` | Архивные артефакты, отделенные от основных таблиц |

Основные ноутбуки:

- [`final_nl2sql_analysis.ipynb`](final_nl2sql_analysis.ipynb)
- [`nl2sql/notebooks/01_report_ea.ipynb`](nl2sql/notebooks/01_report_ea.ipynb)
- [`nl2sql/notebooks/02_report_pass_k.ipynb`](nl2sql/notebooks/02_report_pass_k.ipynb)
- [`nl2sql/notebooks/06_appendix_materials.ipynb`](nl2sql/notebooks/06_appendix_materials.ipynb)
- [`code/notebooks/01_report_fc_passk.ipynb`](code/notebooks/01_report_fc_passk.ipynb)

Пересборка основных ноутбуков:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/01_report_ea.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/02_report_pass_k.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace final_nl2sql_analysis.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/06_appendix_materials.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace code/notebooks/01_report_fc_passk.ipynb
```

## Архивация результатов

Перед новым длительным прогоном можно перенести активные артефакты в архив:

```bash
.venv/bin/python nl2sql/scripts/04_archive_results.py --dry-run
.venv/bin/python nl2sql/scripts/04_archive_results.py --label before_new_run
.venv/bin/python nl2sql/scripts/04_archive_results.py --scope ea --label before_pass_k
```

Архивные каталоги предназначены для хранения промежуточных и исторических
артефактов. Основные таблицы и графики формируются из активных каталогов
`results/nl2sql/` и `results/code/`.
