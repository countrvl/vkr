from pathlib import Path

from shared.config import load_domain_models, load_yaml_config
from code_bench.data.prepare import build_metadata_records, prepare_benchmark_artifacts
from code_bench.evaluation.functional_correctness import evaluate_code_candidate
from code_bench.evaluation.pass_at_k import compute_all_pass_at_k
from code_bench.inference.base import extract_code
from code_bench.data.loader import load_benchmark
from code_bench.prompt.template import PromptBuilder
from shared.inference.api_transport import OpenAIChatTransport


def test_code_domain_layout_exists() -> None:
    assert Path("code/configs/benchmarks.yaml").exists()
    assert Path("code/scripts/01_prepare_benchmarks.py").exists()
    assert Path("code/notebooks/01_report_fc_passk.ipynb").exists()


def test_code_domain_configs_load() -> None:
    benchmarks = load_yaml_config(Path("code/configs/benchmarks.yaml"))
    experiment = load_yaml_config(Path("code/configs/experiment.yaml"))
    models = load_yaml_config(Path("shared/configs/models.yaml"))
    assert benchmarks["data_dir"] == "data/code"
    assert benchmarks["results_dir"] == "results/code/raw"
    assert experiment["k_values"] == [1, 5, 10]
    assert "m1_chatgpt" in models["models"]
    assert "m2_qwen2_5_coder" in models["models"]
    assert models["models"]["m2_qwen2_5_coder"]["supports_code"] is True
    assert models["models"]["m2_qwen2_5_coder"]["supports_sql"] is False


def test_code_domain_model_filter_keeps_only_code_models() -> None:
    models = load_domain_models("supports_code")

    assert "m1_chatgpt" in models
    assert "m2_qwen2_5_coder" in models
    assert "m2_defog" not in models
    assert models["m1_chatgpt"]["max_tokens"] == 768
    assert models["m1_chatgpt"]["prompt_profile"] == "codegen_default"
    assert models["m2_qwen2_5_coder"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen2_5_coder_14b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_deepseek_coder"]["prompt_profile"] == "deepseek_coder"
    assert models["m2_qwen2_5_coder_32b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen3_coder_30b"]["prompt_profile"] == "qwen2_5_coder"
    assert models["m2_qwen2_5_coder_32b"]["active_by_default"] is False
    assert models["m2_qwen3_coder_30b"]["active_by_default"] is False
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
    assert "m2_qwen2_5_coder_32b" not in module._resolve_model_keys("m2", models)
    assert "m2_qwen3_coder_30b" not in module._resolve_model_keys("m2", models)
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
