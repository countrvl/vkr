"""Dataset download helpers for Spider and BIRD."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import httpx


LOGGER = logging.getLogger(__name__)

SPIDER_URL = "https://drive.usercontent.google.com/download?id=1TqleXec_OykOYFREKKtschzY29dUcVAQ&export=download&confirm=t"
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def download_spider(data_dir: Path) -> None:
    """Download and extract the Spider 1.0 dev split.

    Args:
        data_dir: Root directory for benchmark assets.
    """
    target_dir = data_dir / "spider"
    dev_json = target_dir / "dev.json"
    if dev_json.exists():
        LOGGER.info("Spider already present at %s; skipping download.", dev_json)
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="spider_download_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "spider.zip"
        extract_dir = tmp_dir / "extract"
        try:
            LOGGER.info("Downloading Spider dev set from %s", SPIDER_URL)
            _download_file(SPIDER_URL, archive_path)
            _extract_zip(archive_path, extract_dir)
            spider_root = _find_spider_root(extract_dir)
            _copy_tree_contents(spider_root, target_dir)
            _validate_spider_layout(target_dir)
            LOGGER.info("Spider dev set prepared at %s", target_dir)
        except Exception as exc:
            LOGGER.exception("Failed to prepare Spider dev set: %s", exc)
            _log_manual_download_instructions(
                benchmark="Spider",
                source_url=SPIDER_URL,
                expected_layout=(
                    f"{target_dir / 'dev.json'}\n"
                    f"{target_dir / 'tables.json'}\n"
                    f"{target_dir / 'database' / '{db_id}' / '{db_id}.sqlite'}"
                ),
            )
            raise


def download_bird(data_dir: Path) -> None:
    """Download and extract the BIRD dev split.

    Args:
        data_dir: Root directory for benchmark assets.
    """
    target_dir = data_dir / "bird"
    if _bird_dataset_ready(target_dir):
        LOGGER.info("BIRD already present at %s; skipping download.", target_dir)
        return
    if target_dir.exists():
        LOGGER.warning("BIRD directory exists but is incomplete at %s; re-downloading.", target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="bird_download_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "bird_dev.zip"
        extract_dir = tmp_dir / "extract"
        try:
            LOGGER.info("Downloading BIRD dev set from %s", BIRD_URL)
            _download_file(BIRD_URL, archive_path)
            _extract_zip(archive_path, extract_dir)
            _extract_nested_zips(extract_dir)
            bird_root = _find_bird_root(extract_dir)
            _copy_tree_contents(bird_root, target_dir)
            _validate_bird_layout(target_dir)
            LOGGER.info("BIRD dev set prepared at %s", target_dir)
        except Exception as exc:
            LOGGER.exception("Failed to prepare BIRD dev set: %s", exc)
            _log_manual_download_instructions(
                benchmark="BIRD",
                source_url=BIRD_URL,
                expected_layout=(
                    f"{target_dir / 'dev.json'}\n"
                    f"{target_dir / 'dev_databases' / '{db_id}' / '{db_id}.sqlite'}"
                ),
            )
            raise


def _download_file(url: str, destination: Path) -> None:
    """Download a remote file with redirect support.

    Args:
        url: Source URL.
        destination: Output file path.
    """
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)


def _extract_nested_zips(directory: Path) -> None:
    """Extract any ZIP files found inside an already-extracted directory."""
    for nested_zip in directory.rglob("*.zip"):
        dest = nested_zip.parent / nested_zip.stem
        _extract_zip(nested_zip, dest)
        nested_zip.unlink()


def _extract_zip(archive_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive into a directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)


def _find_spider_root(extract_dir: Path) -> Path:
    """Locate the Spider root directory within extracted contents."""
    candidates = [path for path in extract_dir.rglob("dev.json") if path.name == "dev.json"]
    for candidate in candidates:
        parent = candidate.parent
        if (parent / "tables.json").exists() and (parent / "database").exists():
            return parent
        if parent.name == "spider" and (parent / "tables.json").exists():
            return parent
    raise FileNotFoundError("Could not locate Spider root containing dev.json, tables.json, and database/.")


def _find_bird_root(extract_dir: Path) -> Path:
    """Locate the BIRD root directory within extracted contents."""
    candidates = [path.parent for path in extract_dir.rglob("dev.json")]
    for candidate in candidates:
        if (candidate / "dev_databases").exists():
            return candidate
    raise FileNotFoundError("Could not locate BIRD root containing dev.json and dev_databases/.")


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    """Copy all contents from one directory into another."""
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _validate_spider_layout(target_dir: Path) -> None:
    """Validate the normalized Spider directory layout."""
    required_paths: Iterable[Path] = (
        target_dir / "dev.json",
        target_dir / "tables.json",
        target_dir / "database",
    )
    _validate_paths(required_paths, "Spider")


def _validate_bird_layout(target_dir: Path) -> None:
    """Validate the normalized BIRD directory layout."""
    required_paths: Iterable[Path] = (
        target_dir / "dev.json",
        target_dir / "dev_databases",
    )
    _validate_paths(required_paths, "BIRD")
    if not _has_sqlite_files(target_dir / "dev_databases"):
        raise FileNotFoundError(
            f"BIRD extraction incomplete; no SQLite databases found under {target_dir / 'dev_databases'}"
        )
    _normalize_bird_dev_databases(target_dir)


def _bird_dataset_ready(target_dir: Path) -> bool:
    """Return True when the local BIRD dataset is complete enough to use."""
    try:
        _validate_bird_layout(target_dir)
    except FileNotFoundError:
        return False
    return True


def _has_sqlite_files(directory: Path) -> bool:
    """Return True when the directory tree contains at least one SQLite DB."""
    return any(path.is_file() for path in directory.rglob("*.sqlite"))


def _normalize_bird_dev_databases(target_dir: Path) -> None:
    """Flatten the common dev_databases/dev_databases nested extraction layout."""
    root = target_dir / "dev_databases"
    nested = root / "dev_databases"
    if not nested.exists() or not nested.is_dir():
        return

    for item in nested.iterdir():
        destination = root / item.name
        if destination.exists():
            continue
        shutil.move(str(item), str(destination))

    shutil.rmtree(nested)


def _validate_paths(paths: Iterable[Path], benchmark: str) -> None:
    """Ensure required paths exist after extraction."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{benchmark} extraction incomplete; missing: {', '.join(missing)}")


def _log_manual_download_instructions(benchmark: str, source_url: str, expected_layout: str) -> None:
    """Emit a clear manual download fallback message."""
    LOGGER.error(
        "%s download failed. Download manually from %s and place files under:\n%s",
        benchmark,
        source_url,
        expected_layout,
    )
