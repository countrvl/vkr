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

## Актуальные модели

Модели берутся из [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml) и фильтруются по `supports_code: true`.

Основной активный набор `M2`:

| Ключ | Отображаемое имя | Класс | Бэкенд |
| --- | --- | --- | --- |
| `m1_deepseek` | `DeepSeek V3.2` | `M1` | API |
| `m1_chatgpt` | `ChatGPT 5.2` | `M1` | API |
| `m2_qwen2_5_coder` | `Qwen2.5-Coder-7B Instruct Q4_K_M` | `M2` | Ollama |
| `m2_qwen2_5_coder_14b` | `Qwen2.5-Coder-14B Instruct` | `M2` | Ollama |
| `m2_deepseek_coder` | `DeepSeek-Coder-V2-Lite 16B Q4_0` | `M2` | Ollama |

Тяжелые кандидаты для отдельных smoke-run:

- `m2_qwen2_5_coder_32b`
- `m2_qwen3_coder_30b`

Они не входят в default selector `--model m2` и запускаются только по явным ключам.

## Бенчмарки

- `HumanEval+` — benchmark генерации Python-функций с расширенными тестами
- `MBPP+` — benchmark прикладных задач программирования с расширенными тестами

Оценка выполняется через `EvalPlus`.

## Конфигурация

- [`code/configs/benchmarks.yaml`](/home/count/code/vkr/code/configs/benchmarks.yaml)
- [`code/configs/experiment.yaml`](/home/count/code/vkr/code/configs/experiment.yaml)
- [`code/configs/metrics.yaml`](/home/count/code/vkr/code/configs/metrics.yaml)
- [`shared/configs/models.yaml`](/home/count/code/vkr/shared/configs/models.yaml)

Для code-домена prompt profile задается через `domain_overrides.code`.

Текущие профили:

- `m1_deepseek`, `m1_chatgpt` — `codegen_default`
- `m2_qwen2_5_coder`, `m2_qwen2_5_coder_14b`, `m2_qwen2_5_coder_32b`, `m2_qwen3_coder_30b` — `qwen2_5_coder`
- `m2_deepseek_coder` — `deepseek_coder`

Кандидаты `Codestral-22B` и `Devstral-24B` пока зафиксированы только в [`plan.md`](/home/count/code/vkr/plan.md).

## Подготовка данных

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
```

Скрипт подготавливает metadata-артефакты в `data/code/...`.

## Основные команды

### Инференс

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Совместимый alias:

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

### Ноутбук

```bash
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

При необходимости можно указать другой каталог результатов:

```bash
CODE_RESULTS_DIR=/path/to/results/code jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

## Артефакты

- `results/code/raw/*.jsonl`
- `results/code/metrics/fc/*.csv`
- `results/code/metrics/pass_k/*.csv`
- `results/code/figures/fc/*`
- `results/code/figures/pass_k/*`

## Архивация

```bash
.venv/bin/python code/scripts/04_archive_results.py --dry-run
.venv/bin/python code/scripts/04_archive_results.py --label before_new_run
.venv/bin/python code/scripts/04_archive_results.py --scope pass_k --label after_passk
```

Скрипт архивирует текущие `results/code/*` и копирует snapshot code-ноутбука в архив.
