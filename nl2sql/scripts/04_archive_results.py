"""Архивировать текущие артефакты NL2SQL в results/nl2sql/archive/."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = PROJECT_ROOT / "nl2sql"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
_ARTIFACT_DIRS = ("raw", "metrics", "figures")
_RUN_LABELS = ("ea", "pass_k")
_NOTEBOOKS_TO_COPY = ("01_report_ea.ipynb", "02_report_pass_k.ipynb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive current results/{raw,metrics,figures}.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "nl2sql",
        help="Root results directory containing raw/, metrics/, figures/ and archive/.",
    )
    parser.add_argument(
        "--label",
        default="artifacts",
        help="Suffix for the archive directory name. Example: legacy_runs, ea_limit_50.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without changing the filesystem.",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "ea", "pass_k"],
        default="all",
        help="Which experiment artifacts to archive.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _ensure_clean_workdirs(results_dir: Path, dry_run: bool, scope: str) -> None:
    base_dirs = [results_dir / "raw", results_dir / "metrics", results_dir / "figures", results_dir / "archive"]
    scoped_dirs: list[Path] = []
    if scope == "all":
        scoped_dirs.extend((results_dir / "metrics" / label for label in _RUN_LABELS))
        scoped_dirs.extend((results_dir / "figures" / label for label in _RUN_LABELS))
    elif scope in _RUN_LABELS:
        scoped_dirs.append(results_dir / "metrics" / scope)
        scoped_dirs.append(results_dir / "figures" / scope)

    for path in [*base_dirs, *scoped_dirs]:
        if dry_run:
            LOGGER.info("Would ensure directory exists: %s", path)
            continue
        path.mkdir(parents=True, exist_ok=True)


def _move_path(source: Path, destination: Path, dry_run: bool) -> bool:
    if not source.exists():
        LOGGER.info("Skipping missing path: %s", source)
        return False
    if source.is_dir() and not any(source.iterdir()):
        LOGGER.info("Skipping empty directory: %s", source)
        return False

    if dry_run:
        LOGGER.info("Would move %s -> %s", source, destination)
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    LOGGER.info("Moved %s -> %s", source, destination)
    return True


def _archive_paths(results_dir: Path, archive_dir: Path, dry_run: bool, scope: str) -> list[Path]:
    moved: list[Path] = []
    if scope == "all":
        for name in _ARTIFACT_DIRS:
            source = results_dir / name
            destination = archive_dir / name
            if _move_path(source, destination, dry_run):
                moved.append(source)
        return moved

    raw_dir = results_dir / "raw"
    for path in sorted(raw_dir.glob(f"*_{scope}_*.jsonl")):
        destination = archive_dir / "raw" / path.name
        if _move_path(path, destination, dry_run):
            moved.append(path)

    for kind in ("metrics", "figures"):
        source = results_dir / kind / scope
        destination = archive_dir / kind / scope
        if _move_path(source, destination, dry_run):
            moved.append(source)
    return moved


def _copy_report_notebooks(archive_dir: Path, dry_run: bool) -> list[Path]:
    copied: list[Path] = []
    notebooks_dir = DOMAIN_ROOT / "notebooks"
    archive_notebooks_dir = archive_dir / "notebooks"

    for notebook_name in _NOTEBOOKS_TO_COPY:
        source = notebooks_dir / notebook_name
        if not source.exists():
            LOGGER.info("Skipping missing notebook: %s", source)
            continue

        destination = archive_notebooks_dir / notebook_name
        copied.append(source)
        if dry_run:
            LOGGER.info("Would copy %s -> %s", source, destination)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        LOGGER.info("Copied %s -> %s", source, destination)

    return copied


def main() -> None:
    configure_logging(logging.INFO)
    args = parse_args()

    results_dir = args.results_dir.resolve()
    archive_root = results_dir / "archive"
    archive_dir = archive_root / f"{_timestamp()}_{args.label}"

    if archive_dir.exists():
        raise RuntimeError(f"Archive directory already exists: {archive_dir}")

    if args.dry_run:
        LOGGER.info("Dry run: archive destination would be %s", archive_dir)
    else:
        archive_root.mkdir(parents=True, exist_ok=True)

    moved = _archive_paths(results_dir, archive_dir, args.dry_run, args.scope)
    copied = _copy_report_notebooks(archive_dir, args.dry_run)
    if not moved and not copied:
        LOGGER.info("No artifacts found to archive under %s", results_dir)
        _ensure_clean_workdirs(results_dir, args.dry_run, args.scope)
        return

    if args.dry_run:
        LOGGER.info(
            "Would archive %d paths and copy %d notebooks into %s",
            len(moved),
            len(copied),
            archive_dir,
        )
    else:
        _ensure_clean_workdirs(results_dir, dry_run=False, scope=args.scope)
        LOGGER.info(
            "Archived %d paths and copied %d notebooks into %s",
            len(moved),
            len(copied),
            archive_dir,
        )


if __name__ == "__main__":
    main()
