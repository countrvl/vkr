# Code-domain

`code/` - дополнительный экспериментальный домен для сравнения моделей в задаче
генерации Python-кода. Он использует ту же общую инфраструктуру, что и
`nl2sql/`, но имеет отдельные данные, промпты, метрики и результаты.

В текущей версии ВКР основной акцент сделан на NL2SQL. Code-domain оставлен как
самостоятельный воспроизводимый контур и демонстрирует переносимость стенда на
другую задачу.

## Назначение

Домен сравнивает:

- `M1` - крупные general-purpose модели через API;
- `M2` - компактные специализированные модели генерации кода через Ollama.

Основные метрики:

- `Functional Correctness (FC)` - доля задач, в которых сгенерированный код
  проходит тесты;
- `Pass@K` - качество при многократной генерации;
- latency, tokens, cost и производные показатели эффективности.

Оценка выполняется поверх `EvalPlus`.

## Модели

Модели берутся из [`../shared/configs/models.yaml`](../shared/configs/models.yaml)
и фильтруются по `supports_code: true`.

| Ключ | Отображаемое имя | Класс | Backend |
| --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API |
| `m1_claude` | `Claude Sonnet 4.5` | `M1` | Anthropic |
| `m2_qwen2_5_coder` | `Qwen2.5-Coder-7B Instruct Q4_K_M` | `M2` | Ollama |
| `m2_qwen2_5_coder_14b` | `Qwen2.5-Coder-14B Instruct` | `M2` | Ollama |
| `m2_deepseek_coder` | `DeepSeek-Coder-V2-Lite 16B Lite Instruct Q4_0` | `M2` | Ollama |

## Benchmark-и

- `HumanEval+` - генерация Python-функций с расширенными тестами;
- `MBPP+` - прикладные задачи программирования с расширенными тестами.

Данные и metadata-артефакты хранятся в `../data/code/`.

## Структура домена

```text
code/
├── code_bench/    # Импортируемый пакет домена генерации кода
├── configs/       # benchmark, experiment и metrics конфиги
├── notebooks/     # Отчетный ноутбук
├── scripts/       # prepare, inference, evaluation
└── tests/         # Тесты доменного кода
```

Папка `code/` намеренно не является Python-пакетом, чтобы не конфликтовать со
stdlib-модулем `code`. Импортируемый пакет называется `code_bench`.

## Подготовка данных

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
```

Скрипт подготавливает локальные metadata-артефакты в `data/code/...`.

## Запуск inference

Functional Correctness:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
```

Pass@K:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Отдельные модели:

```bash
uv run python code/scripts/02_run_inference.py --model m1_claude --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model m2_qwen2_5_coder_14b --benchmark all --mode fc
```

`ea` поддерживается как alias для `fc`:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode ea
```

Smoke-run:

```bash
uv run python code/scripts/02_run_inference.py --model m1_deepseek --benchmark humaneval_plus --mode fc --limit 5 --mini
```

## Оценка

```bash
uv run python code/scripts/03_evaluate.py --run-label fc
uv run python code/scripts/03_evaluate.py --run-label pass_k
```

После evaluation итоговые таблицы сохраняются в
`results/code/metrics/<run_label>/`.

## Ноутбук

```bash
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

Пересборка без открытия Jupyter:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace code/notebooks/01_report_fc_passk.ipynb
```

## Артефакты

- `results/code/raw/*.jsonl` - raw inference;
- `results/code/metrics/fc/*.csv` - FC-метрики;
- `results/code/metrics/pass_k/*.csv` - Pass@K-метрики, если режим был
  запущен;
- `results/code/figures/fc/*.png` - графики FC;
- `results/code/figures/pass_k/*.png` - графики Pass@K;
- `results/code/archive/` - архивы старых или промежуточных результатов.

`summary_metrics.csv` содержит агрегированные метрики по моделям.
`sample_metrics.csv` и `candidate_metrics.csv` используются для анализа на
уровне отдельных задач и генераций.

## Интерпретация

- Code-domain не смешивается с NL2SQL при формулировке основных выводов ВКР.
- Результаты code-domain можно использовать как дополнительное подтверждение,
  что общая инфраструктура стенда применима к разным доменам.
- Для итоговых сравнений следует использовать только полные прогоны с
  одинаковыми benchmark-ами и режимами.
