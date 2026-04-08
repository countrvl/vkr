# code

Домен для сравнения моделей-кодеров на `HumanEval+` и `MBPP+`.

## Назначение

В домене сравниваются два класса моделей:

- `M1` — крупные general-purpose модели через API
- `M2` — компактные специализированные code-модели через Ollama

Метрики домена:

- `Functional Correctness (FC)`
- `Pass@1`, `Pass@5`, `Pass@10`
- `Efficiency`: latency, tokens, cost

## Модели

Модели берутся из [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) и фильтруются по `supports_code: true`.

| Ключ | Отображаемое имя | Класс | Бэкенд |
| --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API |
| `m1_claude` | `Claude Sonnet 4.5` | `M1` | Anthropic |
| `m2_qwen2_5_coder` | `Qwen2.5-Coder-7B Instruct Q4_K_M` | `M2` | Ollama |
| `m2_qwen2_5_coder_14b` | `Qwen2.5-Coder-14B Instruct` | `M2` | Ollama |
| `m2_deepseek_coder` | `DeepSeek-Coder-V2-Lite 16B Q4_0` | `M2` | Ollama |

## Бенчмарки

- `HumanEval+` — benchmark генерации Python-функций с расширенными тестами
- `MBPP+` — benchmark прикладных задач программирования с расширенными тестами

Оценка выполняется через `EvalPlus`.

## Конфигурация

- [`code/configs/benchmarks.yaml`](/home/count/code/vkr/code/configs/benchmarks.yaml)
- [`code/configs/experiment.yaml`](/home/count/code/vkr/code/configs/experiment.yaml)
- [`code/configs/metrics.yaml`](/home/count/code/vkr/code/configs/metrics.yaml)
- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml)

Для API-моделей ключи доступа читаются из `.env`. Для локальных моделей должен быть доступен `Ollama`.

## Подготовка данных

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
```

Скрипт подготавливает локальные metadata-артефакты в `data/code/...`.

## Основные команды

### Инференс

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Можно запускать и отдельные модели, например:

```bash
uv run python code/scripts/02_run_inference.py --model m1_claude --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model m2_qwen2_5_coder_14b --benchmark all --mode fc
```

`ea` поддерживается как alias для `fc`:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode ea
```

### Smoke-run

```bash
uv run python code/scripts/02_run_inference.py --model m1_deepseek --benchmark humaneval_plus --mode fc --limit 5 --mini
```

### Оценка

```bash
uv run python code/scripts/03_evaluate.py --run-label fc
uv run python code/scripts/03_evaluate.py --run-label pass_k
```

После evaluation итоговые таблицы появляются в `results/code/metrics/<run_label>/`.

### Ноутбук

```bash
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

## Артефакты

- `results/code/raw/*.jsonl`
- `results/code/metrics/fc/*.csv`
- `results/code/metrics/pass_k/*.csv`
- `results/code/figures/fc/*`
- `results/code/figures/pass_k/*`

`summary_metrics.csv` содержит агрегированные метрики по моделям, а `sample_metrics.csv` и `candidate_metrics.csv` позволяют разбирать результаты на уровне отдельных задач и генераций.
