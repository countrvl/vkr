from pathlib import Path

from shared.config import load_yaml_config


def test_code_domain_layout_exists() -> None:
    assert Path("code/configs/benchmarks.yaml").exists()
    assert Path("code/scripts/01_prepare_benchmarks.py").exists()
    assert Path("code/notebooks/01_report_eval.ipynb").exists()


def test_code_domain_configs_load() -> None:
    config = load_yaml_config(Path("code/configs/benchmarks.yaml"))
    assert config["data_dir"] == "data/code"
    assert config["results_dir"] == "results/code/raw"
