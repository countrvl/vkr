import asyncio
import json
from pathlib import Path

from nl2sql.src.data.loader import DataSample
from nl2sql.src.inference.base import GenerationResult
from nl2sql.src.inference.runner import ExperimentRunner


class DummyBackend:
    async def generate(self, *args, **kwargs) -> list[GenerationResult]:
        return [
            GenerationResult(
                sql="SELECT 1",
                raw_response="SELECT 1",
                tokens_input=1,
                tokens_output=1,
                latency_ms=1.0,
                model_name="demo",
                metadata={"backend": "api"},
            )
        ]


class DummyPromptBuilder:
    def build(self, sample: DataSample) -> str:
        return sample.question


def _sample(db_path: Path) -> DataSample:
    return DataSample(
        id="sample-1",
        benchmark="spider",
        question="How many rows?",
        gold_sql="SELECT 1",
        db_id="demo",
        db_path=db_path,
        schema="CREATE TABLE demo(id INTEGER);",
        difficulty="easy",
    )


def test_runner_serializes_db_path_relative_to_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    db_path = data_root / "spider" / "database" / "demo" / "demo.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    runner = ExperimentRunner(
        backend=DummyBackend(),
        prompt_builder=DummyPromptBuilder(),
        output_dir=tmp_path / "out",
        data_root=data_root,
    )

    output_path = asyncio.run(runner.run([_sample(db_path)], model_name="m", benchmark="spider"))
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert payload["db_path"] == "spider/database/demo/demo.sqlite"


def test_runner_falls_back_to_original_db_path_outside_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    outside_db = tmp_path / "external" / "demo.sqlite"
    outside_db.parent.mkdir(parents=True)
    outside_db.touch()

    runner = ExperimentRunner(
        backend=DummyBackend(),
        prompt_builder=DummyPromptBuilder(),
        output_dir=tmp_path / "out",
        data_root=data_root,
    )

    output_path = asyncio.run(runner.run([_sample(outside_db)], model_name="m", benchmark="spider"))
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert payload["db_path"] == str(outside_db)


def test_runner_separates_output_files_by_run_label(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    db_path = data_root / "spider" / "database" / "demo" / "demo.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    runner = ExperimentRunner(
        backend=DummyBackend(),
        prompt_builder=DummyPromptBuilder(),
        output_dir=tmp_path / "out",
        data_root=data_root,
    )

    ea_path = asyncio.run(
        runner.run([_sample(db_path)], model_name="m", benchmark="spider", run_label="ea")
    )
    pass_k_path = asyncio.run(
        runner.run([_sample(db_path)], model_name="m", benchmark="spider", run_label="pass_k")
    )

    assert ea_path != pass_k_path
    assert "_ea_" in ea_path.name
    assert "_pass_k_" in pass_k_path.name


def test_runner_resume_keeps_single_record_and_reuses_existing_progress(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    db_path = data_root / "spider" / "database" / "demo" / "demo.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    runner = ExperimentRunner(
        backend=DummyBackend(),
        prompt_builder=DummyPromptBuilder(),
        output_dir=tmp_path / "out",
        data_root=data_root,
    )
    samples = [_sample(db_path)]

    output_path = asyncio.run(runner.run(samples, model_name="m", benchmark="spider", run_label="ea"))
    second_output_path = asyncio.run(
        runner.run(samples, model_name="m", benchmark="spider", run_label="ea")
    )

    assert output_path == second_output_path
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
