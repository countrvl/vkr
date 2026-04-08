# code

Домен для тестирования моделей-кодеров на `HumanEval+` и `MBPP+`.

Основные метрики:

- `Functional Correctness`
- `Pass@1`, `Pass@5`, `Pass@10`
- efficiency-метрики: latency, tokens, cost

## Конфиги

- [benchmarks.yaml](/home/count/code/vkr/code/configs/benchmarks.yaml)
- [experiment.yaml](/home/count/code/vkr/code/configs/experiment.yaml)
- [shared/configs/models.yaml](/home/count/code/vkr/shared/configs/models.yaml)
- [metrics.yaml](/home/count/code/vkr/code/configs/metrics.yaml)

Модели для code-domain берутся из общего каталога и фильтруются по `supports_code: true`. Доменные runtime-параметры задаются через `domain_overrides.code`.

Для code-domain также задается `prompt_profile`:

- `m1_deepseek`, `m1_chatgpt`: `codegen_default`
- `m2_qwen2_5_coder`: `qwen2_5_coder`
- `m2_qwen2_5_coder_14b`: `qwen2_5_coder`
- `m2_deepseek_coder`: `deepseek_coder`
- `m2_qwen2_5_coder_32b`: `qwen2_5_coder`
- `m2_qwen3_coder_30b`: `qwen2_5_coder`

`CodeGemma-7B` и `CodeLlama-7B` убраны из активного набора `M2`: на smoke-run они были заметно слабее `Qwen2.5-Coder-7B`.
`Codestral-22B` и `Devstral-24B` пока не добавлены в активный каталог; они зафиксированы в `plan.md` как кандидаты для отдельной проверки.

Важно: chat/template-формат модели не собирается вручную в prompt. Для API это делает chat endpoint, а для локальных моделей это делает Ollama Modelfile template. В проекте настраивается только task-level prompt: что просить сгенерировать и какой формат ответа нужен.

## Подготовка данных

```bash
uv run python code/scripts/01_prepare_benchmarks.py --benchmark all
```

Это подтянет `HumanEval+` и `MBPP+` через `EvalPlus` и сохранит локальные metadata-артефакты в `data/code/...`.
Инференс сначала использует подготовленные `metadata.jsonl`/`manifest.json`, если они соответствуют режимам `mini`/`noextreme`; иначе fallback идет напрямую в EvalPlus loader.

## Инференс

Single-shot:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode fc
```

Совместимый alias:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode ea
```

Многократная генерация:

```bash
uv run python code/scripts/02_run_inference.py --model all --benchmark all --mode pass_k
```

Smoke-run:

```bash
uv run python code/scripts/02_run_inference.py --model m1 --benchmark humaneval_plus --mode fc --limit 5 --mini
```

## Оценка

```bash
uv run python code/scripts/03_evaluate.py --run-label fc
uv run python code/scripts/03_evaluate.py --run-label pass_k
```

Артефакты:

- `results/code/raw/*.jsonl`
- `results/code/metrics/fc/summary_metrics.csv`
- `results/code/metrics/fc/sample_metrics.csv`
- `results/code/metrics/fc/candidate_metrics.csv`
- `results/code/metrics/pass_k/summary_metrics.csv`
- `results/code/metrics/pass_k/sample_metrics.csv`
- `results/code/metrics/pass_k/candidate_metrics.csv`

## Ноутбук

```bash
jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

При необходимости выбрать конкретный каталог результатов:

```bash
CODE_RESULTS_DIR=/path/to/results/code jupyter lab code/notebooks/01_report_fc_passk.ipynb
```

## Архивация

```bash
.venv/bin/python code/scripts/04_archive_results.py --dry-run
.venv/bin/python code/scripts/04_archive_results.py --label before_new_run
.venv/bin/python code/scripts/04_archive_results.py --scope pass_k --label after_passk
```
