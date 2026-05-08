# VKR benchmark stand

Репозиторий содержит воспроизводимый экспериментальный стенд для ВКР: сравнение
двух классов языковых моделей в задачах генерации SQL и Python-кода.

Основной исследовательский контур находится в домене [`nl2sql/`](nl2sql/) и
используется для сравнения моделей на `Spider`, `BIRD` и локальном
synthetic e-commerce benchmark (`SEB`). Домен [`code/`](code/) оставлен как
дополнительный независимый контур для генерации кода на `HumanEval+` и
`MBPP+`.

## Назначение проекта

В работе сравниваются два класса моделей:

- `M1` - крупные general-purpose LLM, доступные через API;
- `M2` - компактные специализированные модели, запускаемые локально через
  Ollama.

Проект хранит не только код запуска, но и экспериментальные артефакты:

- raw-ответы моделей;
- агрегированные метрики;
- sample-level метрики для анализа ошибок;
- графики для отчета;
- ноутбуки с итоговыми таблицами и материалами для приложений.

Это позволяет комиссии проверить, как были получены таблицы и графики,
использованные в тексте ВКР.

## Структура репозитория

```text
.
├── nl2sql/                     # Основной домен исследования: Spider, BIRD, SEB
├── code/                       # Дополнительный домен: HumanEval+, MBPP+
├── shared/                     # Общая инфраструктура, конфиги, backends, статистика
├── data/                       # Локальные данные benchmark-ов
├── results/                    # Raw, metrics, figures и архивы результатов
├── final_nl2sql_analysis.ipynb # Итоговый аналитический ноутбук NL2SQL
├── ВКР_v1.md                   # Рабочий текст ВКР
└── pyproject.toml              # Зависимости и настройки проекта
```

## Ключевые результаты и где их смотреть

Основные NL2SQL-артефакты:

- `results/nl2sql/raw/` - активные raw JSONL-файлы для `ea` и `pass_k`;
- `results/nl2sql/metrics/ea/summary_metrics.csv` - сводные метрики
  Execution Accuracy;
- `results/nl2sql/metrics/ea/sample_metrics.csv` - результаты на уровне
  отдельных примеров;
- `results/nl2sql/metrics/pass_k/summary_metrics.csv` - полный актуальный
  Pass@K по основным SQL-моделям;
- `results/nl2sql/figures/ea/` - графики EA;
- `results/nl2sql/figures/pass_k/` - графики Pass@K;
- `results/nl2sql/figures/final/` - итоговые графики для главы 3;
- `results/nl2sql/synthetic_benchmark/` - результаты локального SEB-сценария.

Основные ноутбуки:

- [`final_nl2sql_analysis.ipynb`](final_nl2sql_analysis.ipynb) - итоговый
  анализ NL2SQL;
- [`nl2sql/notebooks/01_report_ea.ipynb`](nl2sql/notebooks/01_report_ea.ipynb)
  - отчет по Execution Accuracy;
- [`nl2sql/notebooks/02_report_pass_k.ipynb`](nl2sql/notebooks/02_report_pass_k.ipynb)
  - отчет по Pass@K;
- [`nl2sql/notebooks/06_appendix_materials.ipynb`](nl2sql/notebooks/06_appendix_materials.ipynb)
  - copy-ready таблицы и графики для приложений ВКР.

## Актуальные NL2SQL-модели

Активное сравнение в основном тексте использует:

| Класс | Ключ | Отображаемое имя | Backend |
| --- | --- | --- | --- |
| `M1` | `m1_deepseek` | `DeepSeek V3.2` | API |
| `M1` | `m1_chatgpt` | `ChatGPT 5.2` | API |
| `M2` | `m2_arctic` | `Arctic-Text2SQL-R1-7B` | Ollama |
| `M2` | `m2_defog` | `Defog-Llama3-SQLCoder-8B q4_0` | Ollama |
| `M2` | `m2_hrida` | `Hrida-T2SQL q8_0` | Ollama |

В `ea` дополнительно присутствует полный результат `Claude Sonnet 4.5`.
Он рассматривается как дополнительный M1-результат, а не как базовый элемент
основной пары `M1`/`M2`.

Старый `SQLCoder-7B` не используется как активный baseline. Неполные и
неактуальные raw-файлы вынесены в `results/nl2sql/archive/`.

## Установка окружения

```bash
uv sync
cp .env.example .env
```

В `.env` задаются ключи API-моделей. Для локальных `M2` должен быть запущен
Ollama и заранее загружены соответствующие модели.

Проверка окружения:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

## Воспроизведение NL2SQL-метрик

Если raw-файлы уже есть, достаточно пересчитать метрики:

```bash
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label ea
.venv/bin/python nl2sql/scripts/03_evaluate.py --run-label pass_k
```

После этого можно пересобрать основные ноутбуки:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/01_report_ea.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/02_report_pass_k.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace final_nl2sql_analysis.ipynb
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace nl2sql/notebooks/06_appendix_materials.ipynb
```

Полный inference может быть длительным, особенно для `BIRD` и `pass_k`.
Команды запуска приведены в [`nl2sql/README.md`](nl2sql/README.md).

## Локальный SEB-сценарий

Synthetic e-commerce benchmark (`SEB`) расположен в
`data/nl2sql/synthetic_ecommerce/`. Он используется как дополнительная
контролируемая SQLite-среда и не заменяет Spider/BIRD.

Основные файлы:

- `dataset_v1.json` - схема и core-запросы;
- `edge_cases_v1.json` - edge-case SQL;
- `snapshot.sqlite` - готовая SQLite-база;
- `coverage_summary.json` - покрытие схемы, запросов и особенностей данных.

Материалы SEB собраны в:

- `nl2sql/notebooks/03_synthetic_ecommerce_report.ipynb`;
- `nl2sql/notebooks/04_synthetic_ecommerce_dataset_description.ipynb`;
- `nl2sql/notebooks/05_synthetic_ecommerce_benchmark_analysis.ipynb`;
- `nl2sql/notebooks/06_appendix_materials.ipynb`.

## Code-domain

Домен [`code/`](code/) содержит отдельный контур для `HumanEval+` и `MBPP+`.
Он полезен как дополнительная демонстрация общей инфраструктуры, но основной
текст текущей ВКР опирается на NL2SQL-результаты.

Команды и артефакты code-domain описаны в [`code/README.md`](code/README.md).

## Архивация и аккуратность результатов

- Актуальные сравнения берутся из `results/nl2sql/raw/`,
  `results/nl2sql/metrics/` и `results/nl2sql/figures/`.
- Архивные результаты лежат в `results/nl2sql/archive/` и не должны
  автоматически смешиваться с текущими таблицами.
- Raw-файлы не удаляются без необходимости; лишние или неполные результаты
  переносятся в архив.
- `expert_scores_template.csv` является шаблоном для экспертной разметки, а не
  результатом экспертного эксперимента.

## Доменные README

- [`nl2sql/README.md`](nl2sql/README.md) - основной домен исследования;
- [`code/README.md`](code/README.md) - дополнительный домен генерации кода.
