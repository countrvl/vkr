from pathlib import Path

import pytest

from src.data import download as download_module


def test_validate_bird_layout_requires_sqlite_files(tmp_path: Path) -> None:
    bird_dir = tmp_path / "bird"
    (bird_dir / "dev_databases").mkdir(parents=True)
    (bird_dir / "dev.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="no SQLite databases found"):
        download_module._validate_bird_layout(bird_dir)


def test_download_bird_redownloads_when_existing_layout_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    target_dir = data_dir / "bird"
    (target_dir / "dev_databases").mkdir(parents=True)
    (target_dir / "dev.json").write_text("[]", encoding="utf-8")

    calls: list[str] = []

    def fake_download_file(url: str, destination: Path) -> None:
        calls.append("download")
        destination.write_bytes(b"placeholder")

    def fake_extract_zip(archive_path: Path, extract_dir: Path) -> None:
        calls.append("extract")
        extract_dir.mkdir(parents=True, exist_ok=True)

    def fake_extract_nested_zips(directory: Path) -> None:
        calls.append("nested")

    def fake_find_bird_root(extract_dir: Path) -> Path:
        root = extract_dir / "bird"
        (root / "dev_databases" / "demo").mkdir(parents=True, exist_ok=True)
        (root / "dev.json").write_text("[]", encoding="utf-8")
        (root / "dev_databases" / "demo" / "demo.sqlite").write_bytes(b"")
        return root

    monkeypatch.setattr(download_module, "_download_file", fake_download_file)
    monkeypatch.setattr(download_module, "_extract_zip", fake_extract_zip)
    monkeypatch.setattr(download_module, "_extract_nested_zips", fake_extract_nested_zips)
    monkeypatch.setattr(download_module, "_find_bird_root", fake_find_bird_root)

    download_module.download_bird(data_dir)

    assert calls == ["download", "extract", "nested"]
    assert (target_dir / "dev_databases" / "demo" / "demo.sqlite").exists()


def test_validate_bird_layout_flattens_nested_dev_databases(tmp_path: Path) -> None:
    bird_dir = tmp_path / "bird"
    nested = bird_dir / "dev_databases" / "dev_databases" / "demo"
    nested.mkdir(parents=True)
    (bird_dir / "dev.json").write_text("[]", encoding="utf-8")
    (nested / "demo.sqlite").write_bytes(b"")

    download_module._validate_bird_layout(bird_dir)

    assert (bird_dir / "dev_databases" / "demo" / "demo.sqlite").exists()
    assert not (bird_dir / "dev_databases" / "dev_databases").exists()
