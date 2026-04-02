from pathlib import Path

from shared.config import load_domain_models, load_yaml_config
from code.src.data.prepare import build_metadata_records, prepare_benchmark_artifacts
from code.src.evaluation.functional_correctness import evaluate_code_candidate
from code.src.evaluation.pass_at_k import compute_all_pass_at_k
from code.src.inference.base import extract_code


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
    assert models["m2_qwen2_5_coder"]["parameters"]["num_ctx"] == 8192


def test_extract_code_from_fenced_response() -> None:
    raw = "Here is the solution:\n```python\ndef foo():\n    return 1\n```"
    assert extract_code(raw) == "def foo():\n    return 1"


def test_prepare_benchmark_artifacts_writes_metadata(tmp_path, monkeypatch) -> None:
    from code.src import data as data_pkg
    from code.src.data import prepare as prepare_module

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
