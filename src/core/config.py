"""Central configuration loading for the audited revision pipeline."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "config.release.yaml"
DEFAULT_LOCAL_CONFIG = PROJECT_ROOT / "configs" / "config.local.yaml"

INPUT_PATH_KEYS = (
    "internal_xr_raw",
    "internal_xr_legacy_processed",
    "bbbd_experiment2_archive",
    "bbbd_experiment3_archive",
    "ds003838_root",
)


class ConfigError(ValueError):
    """Raised when the revision configuration is missing or invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}")

    return data


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))

    for key, value in override.items():
        existing = merged.get(key)

        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def _resolve_paths(
    config: dict[str, Any],
    project_root: Path,
) -> None:
    raw_paths = config.get("paths")

    if not isinstance(raw_paths, Mapping):
        raise ConfigError(
            "The merged configuration must contain a 'paths' mapping. "
            "Create configs/config.local.yaml from the documented example."
        )

    resolved: dict[str, str] = {}

    for key, value in raw_paths.items():
        if value is None or str(value).strip() == "":
            raise ConfigError(f"Empty configured path: paths.{key}")

        path = Path(str(value)).expanduser()

        if not path.is_absolute():
            path = project_root / path

        resolved[key] = str(path.resolve(strict=False))

    config["paths"] = resolved


def _validate_config(
    config: Mapping[str, Any],
    *,
    check_input_paths: bool,
) -> None:
    required_sections = (
        "project",
        "modalities",
        "eeg",
        "preprocessing",
        "features",
        "feature_selection",
        "classifiers",
        "evaluation",
        "datasets",
        "paths",
    )

    missing = [name for name in required_sections if name not in config]

    if missing:
        raise ConfigError(
            "Missing required configuration sections: "
            + ", ".join(missing)
        )

    channels = config["eeg"].get("internal_channels")

    if not isinstance(channels, list) or len(channels) != 8:
        raise ConfigError(
            "eeg.internal_channels must contain exactly eight channels."
        )

    normalized_channels = [str(channel).casefold() for channel in channels]

    if len(set(normalized_channels)) != len(normalized_channels):
        raise ConfigError("eeg.internal_channels contains duplicate channels.")

    internal_primary = (
        config["evaluation"]
        .get("internal", {})
        .get("primary", {})
    )

    if internal_primary.get("group_unit") != "participant":
        raise ConfigError(
            "Internal primary evaluation must group by participant."
        )

    external_within = (
        config["evaluation"]
        .get("external", {})
        .get("within_cohort", {})
    )

    if external_within.get("group_unit") != "subject":
        raise ConfigError(
            "External within-cohort evaluation must group by subject."
        )

    if check_input_paths:
        configured_paths = config["paths"]

        for key in INPUT_PATH_KEYS:
            value = configured_paths.get(key)

            if value is None:
                raise ConfigError(f"Missing required path: paths.{key}")

            path = Path(value)

            if not path.exists():
                raise FileNotFoundError(
                    f"Configured input path does not exist: "
                    f"paths.{key} = {path}"
                )


def load_revision_config(
    base_path: str | Path | None = None,
    local_path: str | Path | None = None,
    *,
    require_local: bool = True,
    check_input_paths: bool = False,
) -> dict[str, Any]:
    """Load, merge, resolve, and validate revision configuration."""

    base = Path(base_path) if base_path else DEFAULT_BASE_CONFIG
    local = Path(local_path) if local_path else DEFAULT_LOCAL_CONFIG

    base = base.expanduser().resolve(strict=False)
    local = local.expanduser().resolve(strict=False)

    config = _load_yaml(base)

    if local.is_file():
        config = _deep_merge(config, _load_yaml(local))
    elif require_local:
        raise FileNotFoundError(
            f"Local configuration not found: {local}. "
            "Create configs/config.local.yaml with machine-specific paths."
        )

    _resolve_paths(config, PROJECT_ROOT)
    _validate_config(config, check_input_paths=check_input_paths)

    config["_meta"] = {
        "project_root": str(PROJECT_ROOT),
        "base_config": str(base),
        "local_config": str(local) if local.is_file() else None,
        "input_paths_checked": check_input_paths,
    }

    return config


def write_resolved_config(
    config: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a resolved configuration manifest as JSON or YAML."""

    destination = Path(output_path)

    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination

    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.suffix.casefold() == ".json":
        destination.write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        destination.write_text(
            yaml.safe_dump(
                dict(config),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    return destination


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and validate the audited revision configuration."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL_CONFIG)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--write", type=Path)

    args = parser.parse_args()

    config = load_revision_config(
        base_path=args.base,
        local_path=args.local,
        check_input_paths=args.check_inputs,
    )

    print(f"Project: {config['project']['name']}")
    print(f"Project root: {config['_meta']['project_root']}")
    print(
        "Internal primary evaluation:",
        config["evaluation"]["internal"]["primary"],
    )
    print(
        "External within-cohort evaluation:",
        config["evaluation"]["external"]["within_cohort"],
    )

    print("\nResolved paths:")
    for key, value in config["paths"].items():
        exists = Path(value).exists()
        print(f"  {key}: {value} [exists={exists}]")

    if args.write:
        destination = write_resolved_config(config, args.write)
        print(f"\nResolved manifest written to: {destination}")


if __name__ == "__main__":
    _main()