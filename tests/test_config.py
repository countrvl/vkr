from pathlib import Path

from src.config import load_yaml_config


def test_models_config_includes_pricing() -> None:
    config = load_yaml_config(Path("configs/models.yaml"))

    assert config["models"]["m1_frontier"]["pricing"]["input_per_1m"] == 0.28
    assert config["models"]["m2_compact"]["pricing"]["output_per_1m"] == 0.0


def test_metrics_config_includes_statistical_tests_and_pricing() -> None:
    config = load_yaml_config(Path("configs/metrics.yaml"))

    assert config["deepseek_pricing"]["output_per_1m"] == 0.42
    assert config["ollama_pricing"]["input_per_1m"] == 0.0
    assert config["statistical_tests"]["alpha"] == 0.05


def test_experiment_config_includes_path_defaults() -> None:
    config = load_yaml_config(Path("configs/experiment.yaml"))

    assert config["data_dir"] == "data"
    assert config["results_dir"] == "results/raw"
