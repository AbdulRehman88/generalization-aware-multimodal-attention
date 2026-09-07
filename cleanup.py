"""Safely identify or remove local Python and IDE cache artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DIRECTORIES = {".idea", "__pycache__"}
TARGET_SUFFIXES = {".pyc", ".pyo"}


def remove_unwanted_files(
    project_root: str | Path = PROJECT_ROOT,
    *,
    apply: bool = False,
) -> list[Path]:
    """Find cache artifacts and remove them only when ``apply`` is true."""

    root = Path(project_root).expanduser().resolve(strict=True)

    if not (root / ".git").exists():
        raise ValueError(
            "Cleanup is restricted to a Git repository root: "
            f"{root}"
        )

    targets: list[Path] = []

    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)

        directory_names[:] = [
            name for name in directory_names
            if name != ".git"
        ]

        removable_directories = [
            name for name in directory_names
            if name in TARGET_DIRECTORIES
        ]

        for name in removable_directories:
            targets.append(current / name)
            directory_names.remove(name)

        for name in file_names:
            candidate = current / name
            if candidate.suffix.casefold() in TARGET_SUFFIXES:
                targets.append(candidate)

    targets = sorted(set(targets), key=lambda item: str(item).casefold())

    print(f"Repository root: {root}")
    print(f"Mode: {'APPLY' if apply else 'DRY RUN'}")
    print(f"Targets found: {len(targets)}")

    for target in targets:
        print(f"  {target.relative_to(root)}")

    if apply:
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

        print(f"Removed: {len(targets)}")
    else:
        print("Nothing removed. Use --apply to perform cleanup.")

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find repository-local .idea, __pycache__, "
            ".pyc, and .pyo artifacts."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Git repository root; defaults to this script's directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform removal. Without this flag, only a dry run is shown.",
    )
    arguments = parser.parse_args()

    remove_unwanted_files(
        arguments.root,
        apply=arguments.apply,
    )


if __name__ == "__main__":
    main()
