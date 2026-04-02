from pathlib import Path
import importlib.util

from shared.config import load_domain_models, load_yaml_config


_INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "script_02_run_inference",
    Path(__file__).resolve().parents[1] / "scripts" / "02_run_inference.py",
)
assert _INFERENCE_SPEC is not None and _INFERENCE_SPEC.loader is not None
_INFERENCE_MODULE = importlib.util.module_from_spec(_INFERENCE_SPEC)
_INFERENCE_SPEC.loader.exec_module(_INFERENCE_MODULE)


def test_models_config_includes_pricing() -> None:
    config = load_yaml_config(Path("shared/configs/models.yaml"))

    assert config["models"]["m1_deepseek"]["pricing"]["input_per_1m"] == 0.28
    assert config["models"]["m1_deepseek"]["supports_sql"] is True
    assert config["models"]["m1_deepseek"]["supports_code"] is True
    assert config["models"]["m1_deepseek"]["base_url_env"] == "DEEPSEEK_API_URL"
    assert config["models"]["m1_deepseek"]["model_id_env"] == "DEEPSEEK_MODEL_ID"
    assert config["models"]["m1_deepseek"]["name"] == "DeepSeek"
    assert config["models"]["m1_chatgpt"]["model_id"] == "openai/gpt-5.2"
    assert config["models"]["m1_chatgpt"]["base_url_env"] == "OPENAI_API_URL"
    assert config["models"]["m1_chatgpt"]["model_id_env"] == "OPENAI_MODEL_ID"
    assert config["models"]["m1_chatgpt"]["env_key"] == "OPENAI_API_KEY"
    assert config["models"]["m1_chatgpt"]["name"] == "ChatGPT"
    assert config["models"]["m1_chatgpt"]["pricing"]["output_per_1m"] == 10.0
    assert config["models"]["m2_defog"]["pricing"]["output_per_1m"] == 0.0
    assert config["models"]["m2_defog"]["base_url_env"] == "OLLAMA_API_URL"
    assert config["models"]["m2_defog"]["supports_sql"] is True
    assert config["models"]["m2_defog"]["supports_code"] is False
    assert config["models"]["m2_hrida"]["model_id"] == "HridaAI/hrida-t2sql:q8_0"
    assert config["models"]["m2_hrida"]["pricing"]["output_per_1m"] == 0.0
    assert config["models"]["m2_hrida"]["base_url_env"] == "OLLAMA_API_URL"
    assert config["models"]["m2_arctic"]["model_id"] == "a-kore/Arctic-Text2SQL-R1-7B:latest"
    assert config["models"]["m2_arctic"]["pricing"]["output_per_1m"] == 0.0
    assert config["models"]["m2_arctic"]["base_url_env"] == "OLLAMA_API_URL"
    assert config["models"]["m2_qwen2_5_coder"]["supports_sql"] is False
    assert config["models"]["m2_qwen2_5_coder"]["supports_code"] is True


def test_metrics_config_includes_statistical_tests_and_pricing() -> None:
    config = load_yaml_config(Path("nl2sql/configs/metrics.yaml"))

    assert config["deepseek_pricing"]["output_per_1m"] == 0.42
    assert config["ollama_pricing"]["input_per_1m"] == 0.0
    assert config["statistical_tests"]["alpha"] == 0.05


def test_domain_model_filter_keeps_only_sql_models() -> None:
    models = load_domain_models("supports_sql")

    assert "m1_deepseek" in models
    assert "m2_defog" in models
    assert "m2_qwen2_5_coder" not in models
    assert models["m1_deepseek"]["max_tokens"] == 512
    assert models["m2_defog"]["parameters"]["num_ctx"] == 4096


def test_experiment_config_includes_path_defaults() -> None:
    config = load_yaml_config(Path("nl2sql/configs/experiment.yaml"))

    assert config["data_dir"] == "data/nl2sql"
    assert config["results_dir"] == "results/nl2sql/raw"


def test_build_backend_reads_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_URL", "https://override.example/api")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    backend = _INFERENCE_MODULE._build_backend(
        "m1_deepseek",
        {
            "backend": "api",
            "model_id": "demo-model",
            "base_url": "https://default.example/api",
            "base_url_env": "DEEPSEEK_API_URL",
            "env_key": "DEEPSEEK_API_KEY",
            "name": "Demo",
            "parameters": {},
        },
    )

    assert str(backend.client.base_url) == "https://override.example/api/"


def test_build_backend_reads_model_id_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_ID", "openai/custom-model")

    backend = _INFERENCE_MODULE._build_backend(
        "m1_chatgpt",
        {
            "backend": "api",
            "model_id": "openai/default-model",
            "model_id_env": "OPENAI_MODEL_ID",
            "base_url": "https://default.example/api",
            "env_key": "OPENAI_API_KEY",
            "name": "ChatGPT",
            "parameters": {},
        },
    )

    assert backend.model_id == "openai/custom-model"


def test_resolve_model_keys_supports_groups_and_lists() -> None:
    models_cfg = {
        "m1_deepseek": {},
        "m1_chatgpt": {},
        "m2_defog": {},
        "m2_hrida": {},
    }

    assert _INFERENCE_MODULE._resolve_model_keys("all", models_cfg) == [
        "m1_deepseek",
        "m1_chatgpt",
        "m2_defog",
        "m2_hrida",
    ]
    assert _INFERENCE_MODULE._resolve_model_keys("m1", models_cfg) == [
        "m1_deepseek",
        "m1_chatgpt",
    ]
    assert _INFERENCE_MODULE._resolve_model_keys("m2", models_cfg) == [
        "m2_defog",
        "m2_hrida",
    ]
    assert _INFERENCE_MODULE._resolve_model_keys("m1,m2_defog", models_cfg) == [
        "m1_deepseek",
        "m1_chatgpt",
        "m2_defog",
    ]
    assert _INFERENCE_MODULE._resolve_model_keys(
        "m1_deepseek,m2_defog,m1_deepseek",
        models_cfg,
    ) == ["m1_deepseek", "m2_defog"]


def test_resolve_model_keys_rejects_unknown_selector() -> None:
    models_cfg = {"m1_deepseek": {}, "m2_defog": {}}

    try:
        _INFERENCE_MODULE._resolve_model_keys("m3_unknown", models_cfg)
    except ValueError as exc:
        assert "Unknown model selector" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model selector")


def test_mode_configuration_disables_seed_for_pass_k() -> None:
    exp_cfg = {
        "seed": 42,
        "temperature_pass_k": 0.8,
        "k_values": [1, 5, 10],
    }

    assert _INFERENCE_MODULE._resolve_mode_params("ea", exp_cfg) == (0.0, 1, 42)
    assert _INFERENCE_MODULE._resolve_mode_params("pass_k", exp_cfg) == (0.8, 10, None)
