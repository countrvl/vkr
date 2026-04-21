import asyncio
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from shared.config import load_domain_models, load_yaml_config
from code_bench.data.prepare import build_metadata_records, prepare_benchmark_artifacts
from code_bench.evaluation.functional_correctness import evaluate_code_candidate
from code_bench.evaluation.pass_at_k import compute_all_pass_at_k
from code_bench.inference.base import extract_code
from code_bench.data.loader import load_benchmark
from code_bench.inference.runner import ExperimentRunner
from code_bench.prompt.template import PromptBuilder
from shared.inference.api_transport import OpenAIChatTransport
from shared.inference.anthropic_transport import AnthropicMessagesTransport


_EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "code_script_03_evaluate",
    Path(__file__).resolve().parents[1] / "scripts" / "03_evaluate.py",
)
assert _EVALUATE_SPEC is not None and _EVALUATE_SPEC.loader is not None
_EVALUATE_MODULE = importlib.util.module_from_spec(_EVALUATE_SPEC)
_EVALUATE_SPEC.loader.exec_module(_EVALUATE_MODULE)


def test_code_domain_layout_exists() -> None:
    assert Path("code/configs/benchmarks.yaml").exists()
    assert Path("code/scripts/01_prepare_benchmarks.py").exists()
    assert Path("code/notebooks/01_report_fc_passk.ipynb").exists()


def test_code_domain_configs_load() -> None:
    benchmarks = load_yaml_config(Path("code/configs/benchmarks.yaml"))
    experiment = load_yaml_config(Path("code/configs/experiment.yaml"))
    metrics = load_yaml_config(Path("code/configs/metrics.yaml"))
    models = load_yaml_config(Path("shared/configs/models.yaml"))
    assert benchmarks["data_dir"] == "data/code"
    assert benchmarks["results_dir"] == "results/code/raw"
    assert experiment["k_values"] == [1, 5, 10]
    assert "m1_chatgpt" in models["models"]
    assert "m1_claude" in models["models"]
    assert "m1_qwen3_6_plus" in models["models"]
    assert "m2_qwen2_5_coder" in models["models"]
    assert models["models"]["m1_claude"]["batch_support"] is True
    assert models["models"]["m1_chatgpt"]["batch_support"] is False
    assert models["models"]["m1_qwen3_6_plus"]["supports_code"] is True
    assert models["models"]["m1_qwen3_6_plus"]["supports_sql"] is True
    assert models["models"]["m2_qwen2_5_coder"]["supports_code"] is True
    assert models["models"]["m2_qwen2_5_coder"]["supports_sql"] is False
    assert metrics["statistics"]["quantiles"] == [0.05, 0.5, 0.95]


def test_code_domain_model_filter_keeps_only_code_models() -> None:
    models = load_domain_models("supports_code")

    assert "m1_chatgpt" in models
    assert "m1_claude" in models
    assert "m1_qwen3_6_plus" in models
    assert "m2_qwen2_5_coder" in models
    assert "m2_defog" not in models
    assert models["m1_chatgpt"]["max_tokens"] == 768
    assert models["m1_chatgpt"]["prompt_profile"] == "codegen_default"
    assert models["m1_qwen3_6_plus"]["prompt_profile"] == "codegen_default"
    assert models["m2_qwen2_5_coder"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen2_5_coder_14b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_deepseek_coder"]["prompt_profile"] == "deepseek_coder"
    assert models["m2_qwen2_5_coder_32b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen3_coder_30b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen2_5_coder_32b"]["active_by_default"] is False
    assert models["m2_qwen3_coder_30b"]["active_by_default"] is False
    assert models["m1_qwen3_6_plus"]["active_by_default"] is False
    assert "m2_codegemma" not in models
    assert "m2_codellama" not in models
    assert models["m2_qwen2_5_coder"]["parameters"]["num_ctx"] == 8192


def test_code_model_selector_excludes_heavy_candidates_by_default() -> None:
    from importlib import util

    script_path = Path("code/scripts/02_run_inference.py")
    spec = util.spec_from_file_location("code_run_inference", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    models = load_domain_models("supports_code")
    assert "m1_qwen3_6_plus" not in module._resolve_model_keys("m1", models)
    assert "m2_qwen2_5_coder_32b" not in module._resolve_model_keys("m2", models)
    assert "m2_qwen3_coder_30b" not in module._resolve_model_keys("m2", models)
    assert module._resolve_model_keys("m1_qwen3_6_plus", models) == ["m1_qwen3_6_plus"]
    assert module._resolve_model_keys("m2_qwen2_5_coder_32b", models) == [
        "m2_qwen2_5_coder_32b"
    ]


def test_extract_code_from_fenced_response() -> None:
    raw = "Here is the solution:\n```python\ndef foo():\n    return 1\n```"
    assert extract_code(raw) == "def foo():\n    return 1"


def test_code_prompt_profiles_render() -> None:
    from code_bench.data.schema import CodeSample

    sample = CodeSample(
        id="Task/1",
        benchmark="humaneval_plus",
        prompt_text='def solve(x):\n    """Return x."""\n',
        entry_point="solve",
        canonical_solution="",
        contract="",
    )
    builder = PromptBuilder()
    for profile in (
        "codegen_default",
        "qwen2_5_coder",
        "deepseek_coder",
    ):
        prompt = builder.build(sample, prompt_profile=profile)
        assert "solve" in prompt
        assert "```" not in prompt


def test_api_transport_rejects_placeholder_response() -> None:
    transport = OpenAIChatTransport(
        model_id="demo",
        base_url="http://example.invalid",
        api_key="test",
        model_name="demo",
        extractor=lambda raw: raw,
        result_factory=lambda **kwargs: kwargs,
    )

    try:
        transport._validate_response_content("no assistant response")
    except RuntimeError as exc:
        assert "invalid placeholder" in str(exc)
    else:
        raise AssertionError("Expected placeholder response to be rejected.")


def test_anthropic_transport_parses_batch_results() -> None:
    transport = AnthropicMessagesTransport(
        model_id="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        api_key="test",
        model_name="Claude",
        parameters={"batch_pricing_multiplier": 0.5},
        pricing={"input_per_1m": 3.0, "output_per_1m": 15.0},
        extractor=lambda raw: raw.strip(),
        result_factory=lambda **kwargs: kwargs,
        use_batch=True,
    )

    async def fake_request_json(method: str, path: str, *, json_payload=None):
        if method == "POST":
            return {"id": "msgbatch_123", "processing_status": "in_progress"}
        return {"id": "msgbatch_123", "processing_status": "ended", "results_url": "https://example.test/results"}

    async def fake_request_jsonl(url: str):
        assert url == "https://example.test/results"
        return [
            {
                "custom_id": "req_000000",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "print('ok')"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                },
            }
        ]

    transport._request_json = fake_request_json
    transport._request_jsonl = fake_request_jsonl

    results = asyncio.run(
        transport.generate_batch(
            prompts=["write code"],
            temperature=0.0,
            max_tokens=64,
            seed=None,
            top_p=None,
        )
    )
    item = results[0]
    assert not isinstance(item, Exception)
    generation = item[0]
    assert generation["raw_response"] == "print('ok')"
    assert generation["tokens_input"] == 10
    assert generation["tokens_output"] == 5
    assert generation["metadata"]["dispatch"] == "batch"
    assert generation["metadata"]["batch_id"] == "msgbatch_123"
    assert generation["metadata"]["pricing_multiplier"] == 0.5


def test_anthropic_transport_emits_status_callbacks() -> None:
    transport = AnthropicMessagesTransport(
        model_id="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        api_key="test",
        model_name="Claude",
        extractor=lambda raw: raw.strip(),
        result_factory=lambda **kwargs: kwargs,
        use_batch=True,
    )
    events: list[dict] = []
    transport.status_callback = events.append

    async def fake_request_json(method: str, path: str, *, json_payload=None):
        if method == "POST":
            return {"id": "msgbatch_456", "processing_status": "in_progress"}
        return {
            "id": "msgbatch_456",
            "processing_status": "ended",
            "results_url": "https://example.test/results",
        }

    async def fake_request_jsonl(url: str):
        return [
            {
                "custom_id": "req_000000",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "hello"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            }
        ]

    transport._request_json = fake_request_json
    transport._request_jsonl = fake_request_jsonl

    asyncio.run(
        transport.generate_batch(
            prompts=["ping"],
            temperature=0.0,
            max_tokens=16,
            seed=None,
            top_p=None,
        )
    )

    statuses = [event["status"] for event in events]
    assert "in_progress" in statuses
    assert "ended" in statuses
    assert "completed" in statuses


def test_anthropic_transport_can_resume_existing_batch() -> None:
    transport = AnthropicMessagesTransport(
        model_id="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        api_key="test",
        model_name="Claude",
        extractor=lambda raw: raw.strip(),
        result_factory=lambda **kwargs: kwargs,
        use_batch=True,
    )

    async def fake_request_json(method: str, path: str, *, json_payload=None):
        assert method == "GET"
        assert path == "/v1/messages/batches/msgbatch_resume"
        return {
            "id": "msgbatch_resume",
            "processing_status": "ended",
            "created_at": "2026-04-08T19:20:58.117962+00:00",
            "results_url": "https://example.test/results",
        }

    async def fake_request_jsonl(url: str):
        assert url == "https://example.test/results"
        return [
            {
                "custom_id": "req_000000",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": "hello"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
            }
        ]

    transport._request_json = fake_request_json
    transport._request_jsonl = fake_request_jsonl

    results = asyncio.run(
        transport.resume_batch(
            batch_id="msgbatch_resume",
            prompts=["ping"],
            temperature=0.0,
            max_tokens=16,
            seed=None,
            top_p=None,
        )
    )

    item = results[0]
    assert not isinstance(item, Exception)
    assert item[0]["metadata"]["batch_id"] == "msgbatch_resume"


def test_anthropic_transport_does_not_fallback_after_submit() -> None:
    transport = AnthropicMessagesTransport(
        model_id="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com",
        api_key="test",
        model_name="Claude",
        extractor=lambda raw: raw.strip(),
        result_factory=lambda **kwargs: kwargs,
        use_batch=True,
    )
    online_called = False

    async def fake_online_many(prompts, temperature, max_tokens, seed, top_p):
        nonlocal online_called
        online_called = True
        return []

    async def fake_request_json(method: str, path: str, *, json_payload=None):
        if method == "POST":
            return {"id": "msgbatch_submit", "processing_status": "in_progress"}
        raise RuntimeError("poll failed")

    transport._generate_online_many = fake_online_many
    transport._request_json = fake_request_json

    try:
        asyncio.run(
            transport.generate_batch(
                prompts=["ping"],
                temperature=0.0,
                max_tokens=16,
                seed=None,
                top_p=None,
            )
        )
    except RuntimeError as exc:
        assert "poll failed" in str(exc)
    else:
        raise AssertionError("Expected polling error to propagate after batch submit.")

    assert online_called is False


def test_batch_manifest_updates_preserve_batch_metadata(tmp_path) -> None:
    class DummyBackend:
        supports_batch = False

    runner = ExperimentRunner(DummyBackend(), PromptBuilder(), tmp_path / "raw")
    output_path = tmp_path / "raw" / "Claude_humaneval_plus_fc_20260408_000000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = runner._manifest_path(output_path)

    runner._write_batch_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        benchmark="humaneval_plus",
        model_key="m1_claude",
        model_name="Claude",
        run_label="fc",
        n_requests=5,
        status="polling",
        batch_id="msgbatch_123",
        phase="polling",
        elapsed_seconds=12.5,
    )
    runner._write_batch_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        benchmark="humaneval_plus",
        model_key="m1_claude",
        model_name="Claude",
        run_label="fc",
        n_requests=5,
        status="completed",
        phase="completed",
    )

    payload = load_yaml_config(manifest_path)
    assert payload["batch_id"] == "msgbatch_123"
    assert payload["status"] == "completed"
    assert payload["phase"] == "completed"
    assert payload["n_requests"] == 5
    assert "created_at" in payload
    assert "updated_at" in payload


def test_runner_detects_resumable_batch_manifest(tmp_path) -> None:
    class DummyBackend:
        supports_batch = False

    runner = ExperimentRunner(DummyBackend(), PromptBuilder(), tmp_path / "raw")
    output_path = tmp_path / "raw" / "Claude_humaneval_plus_fc_20260408_000000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = runner._manifest_path(output_path)
    manifest_path.write_text(
        """{
  "model_key": "m1_claude",
  "model_name": "Claude",
  "benchmark": "humaneval_plus",
  "run_label": "fc",
  "raw_output_path": "%s",
  "status": "in_progress",
  "batch_id": "msgbatch_resume"
}"""
        % output_path,
        encoding="utf-8",
    )

    batch_id = runner._resumable_batch_id(
        manifest_path=manifest_path,
        output_path=output_path,
        benchmark="humaneval_plus",
        model_name="Claude",
        run_label="fc",
        completed_ids=set(),
    )

    assert batch_id == "msgbatch_resume"


def test_prepare_benchmark_artifacts_writes_metadata(tmp_path, monkeypatch) -> None:
    from code_bench.data import prepare as prepare_module

    monkeypatch.setattr(
        prepare_module,
        "load_evalplus_tasks",
        lambda benchmark, mini=False, noextreme=False: {
            "Task/1": {
                "task_id": "Task/1",
                "entry_point": "solve",
                "prompt": "def solve(x):\n    pass\n",
                "canonical_solution": "\n    return x\n",
                "contract": "",
                "atol": 0,
                "base_input": [[1]],
                "plus_input": [[2]],
            }
        },
    )
    monkeypatch.setattr(prepare_module, "get_benchmark_hash", lambda benchmark, mini=False, noextreme=False: "hash123")
    manifest = prepare_benchmark_artifacts(
        "humaneval_plus",
        data_dir=tmp_path,
        local_dir=tmp_path / "humaneval_plus",
        mini=False,
    )
    assert manifest["dataset_hash"] == "hash123"
    assert (tmp_path / "humaneval_plus" / "metadata.jsonl").exists()
    assert (tmp_path / "humaneval_plus" / "manifest.json").exists()
    samples = load_benchmark("humaneval_plus", tmp_path)
    assert samples[0].id == "Task/1"
    assert samples[0].metadata["source"] == "prepared_metadata"


def test_build_metadata_records() -> None:
    rows = build_metadata_records(
        "humaneval_plus",
        {
            "Task/1": {
                "task_id": "Task/1",
                "entry_point": "solve",
                "prompt": "def solve(x):\n    pass\n",
                "canonical_solution": "\n    return x\n",
                "contract": "",
                "atol": 0,
                "base_input": [[1]],
                "plus_input": [[2]],
            }
        },
    )
    assert rows[0]["sample_id"] == "Task/1"
    assert rows[0]["n_base_tests"] == 1


def test_evaluate_code_candidate_canonical_solution_passes() -> None:
    result = evaluate_code_candidate(
        benchmark="humaneval_plus",
        task_id="HumanEval/0",
        candidate_index=0,
        code="def has_close_elements(numbers, threshold):\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n",
        execution_cfg={"fast_check": False, "min_time_limit": 1.0, "gt_time_limit_factor": 4.0},
    )
    assert result["compiled_ok"] is True
    assert result["functional_correctness"] is True


def test_evaluate_code_candidate_syntax_error() -> None:
    result = evaluate_code_candidate(
        benchmark="humaneval_plus",
        task_id="HumanEval/0",
        candidate_index=0,
        code="def has_close_elements(:\n    pass\n",
        execution_cfg={"fast_check": False, "min_time_limit": 1.0, "gt_time_limit_factor": 4.0},
    )
    assert result["compiled_ok"] is False
    assert result["error_type"] == "syntax_error"


def test_code_pass_at_k_uses_prefix_semantics() -> None:
    metrics = compute_all_pass_at_k([[False, True, False]], [1, 5, 10])
    assert metrics[1] == 0.0
    assert metrics[5] == 1.0
    assert metrics[10] == 1.0


def test_code_evaluate_writes_summary_statistics(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"
    raw_dir.mkdir()

    record = {
        "sample_id": "HumanEval/0",
        "model_key": "m1_demo",
        "model_name": "Demo",
        "model_display_name": "Demo",
        "model_version": "1.0",
        "benchmark": "humaneval_plus",
        "run_label": "fc",
        "entry_point": "has_close_elements",
        "prompt": "def has_close_elements(numbers, threshold):\n    pass\n",
        "generations": [
            {
                "code": "def has_close_elements(numbers, threshold):\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n",
                "raw_response": "ok",
                "tokens_input": 20,
                "tokens_output": 10,
                "latency_ms": 25.0,
                "model_name": "Demo",
                "metadata": {"backend": "ollama"},
            }
        ],
    }
    (raw_dir / "demo.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        _EVALUATE_MODULE,
        "parse_args",
        lambda: SimpleNamespace(
            config_dir=Path("code/configs"),
            raw_dir=raw_dir,
            output_dir=output_dir,
            run_label="all",
        ),
    )

    _EVALUATE_MODULE.main()

    summary_path = output_dir / "fc" / "summary_metrics.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["functional_correctness"] == "1.0"
    assert rows[0]["functional_correctness_ci_low"]
    assert rows[0]["functional_correctness_ci_high"]
    assert rows[0]["functional_correctness_q05"] == "1.0"
    assert rows[0]["functional_correctness_q50"] == "1.0"
    assert rows[0]["functional_correctness_q95"] == "1.0"
    assert rows[0]["Tinf_q50"] == "25.0"
    assert rows[0]["Tok_q50"] == "30.0"
    assert rows[0]["Cost_q50"] == "0.0"
    assert rows[0]["pass@1_q05"] == "1.0"
    assert rows[0]["pass@1_q50"] == "1.0"
    assert rows[0]["pass@1_q95"] == "1.0"
    assert rows[0]["pass@1_ci_low"]
    assert rows[0]["pass@1_ci_high"]


def test_code_analysis_filters_main_vs_appendix() -> None:
    from code_analysis_utils import build_completeness_audit, filter_appendix_rows, filter_main_report_rows

    summary_df = pd.DataFrame(
        [
            {
                "model_key": "m1_chatgpt",
                "model_name": "ChatGPT",
                "model_display_name": "ChatGPT 5.2",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "n_samples": 164,
            },
            {
                "model_key": "m1_chatgpt",
                "model_name": "ChatGPT",
                "model_display_name": "ChatGPT 5.2",
                "benchmark": "mbpp_plus",
                "run_label": "fc",
                "n_samples": 378,
            },
            {
                "model_key": "m1_qwen3_6_plus",
                "model_name": "Qwen 3.6 Plus",
                "model_display_name": "Qwen 3.6 Plus",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "n_samples": 164,
            },
            {
                "model_key": "m1_qwen3_6_plus",
                "model_name": "Qwen 3.6 Plus",
                "model_display_name": "Qwen 3.6 Plus",
                "benchmark": "mbpp_plus",
                "run_label": "fc",
                "n_samples": 378,
            },
            {
                "model_key": "m2_qwen2_5_coder",
                "model_name": "Qwen2.5-Coder-7B",
                "model_display_name": "Qwen2.5-Coder-7B Instruct Q4_K_M",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "n_samples": 164,
            },
        ]
    )

    audit_df = build_completeness_audit(summary_df, run_label="fc")
    main_df = filter_main_report_rows(summary_df, run_label="fc", audit_df=audit_df)
    appendix_df = filter_appendix_rows(summary_df, run_label="fc", audit_df=audit_df)

    assert set(audit_df.loc[audit_df["is_complete_main"], "model_key"]) == {
        "m1_chatgpt",
        "m1_qwen3_6_plus",
    }
    assert set(main_df["model_key"]) == {"m1_chatgpt"}
    assert set(appendix_df["model_key"]) == {"m1_qwen3_6_plus", "m2_qwen2_5_coder"}


def test_code_analysis_pairwise_fc_deltas() -> None:
    from code_analysis_utils import compute_pairwise_fc_deltas

    sample_df = pd.DataFrame(
        [
            {
                "sample_id": "Task/1",
                "model_key": "m1_chatgpt",
                "model_name": "ChatGPT",
                "model_display_name": "ChatGPT 5.2",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "first_hit": True,
            },
            {
                "sample_id": "Task/2",
                "model_key": "m1_chatgpt",
                "model_name": "ChatGPT",
                "model_display_name": "ChatGPT 5.2",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "first_hit": False,
            },
            {
                "sample_id": "Task/1",
                "model_key": "m2_qwen2_5_coder",
                "model_name": "Qwen2.5-Coder-7B",
                "model_display_name": "Qwen2.5-Coder-7B Instruct Q4_K_M",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "first_hit": False,
            },
            {
                "sample_id": "Task/2",
                "model_key": "m2_qwen2_5_coder",
                "model_name": "Qwen2.5-Coder-7B",
                "model_display_name": "Qwen2.5-Coder-7B Instruct Q4_K_M",
                "benchmark": "humaneval_plus",
                "run_label": "fc",
                "first_hit": False,
            },
        ]
    )

    deltas = compute_pairwise_fc_deltas(sample_df)

    assert len(deltas) == 1
    assert deltas.loc[0, "benchmark"] == "humaneval_plus"
    assert deltas.loc[0, "n_pairs"] == 2
    assert deltas.loc[0, "delta"] == 0.5
    assert deltas.loc[0, "delta_q05"] <= deltas.loc[0, "delta_q50"] <= deltas.loc[0, "delta_q95"]
