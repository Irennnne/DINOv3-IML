#!/usr/bin/env python3
"""
Resolve YAML experiment configs with optional base config inheritance.

Supported keys:
  - base_config: "relative/path/to/base.yaml"
  - base_configs:
      - "relative/path/to/base1.yaml"
      - "relative/path/to/base2.yaml"
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_entries(config: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    if "base_config" in config:
        entries.append(config["base_config"])
    if "base_configs" in config:
        values = config["base_configs"]
        if isinstance(values, str):
            entries.append(values)
        else:
            entries.extend(values)
    return entries


def load_config(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Detected config inheritance cycle: {cycle}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    merged: dict[str, Any] = {}
    for entry in _base_entries(config):
        base_path = (path.parent / entry).resolve()
        merged = _deep_merge(merged, load_config(base_path, (*stack, path)))

    config.pop("base_config", None)
    config.pop("base_configs", None)
    return _deep_merge(merged, config)


def to_cli_args(config: dict[str, Any], skip: set[str]) -> str:
    parts: list[str] = []
    for key, value in config.items():
        if key in skip:
            continue
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
        else:
            parts.append(f"--{key} {shlex.quote(str(value))}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve inherited YAML configs")
    parser.add_argument("config", help="Path to the top-level YAML config")
    parser.add_argument("--field", help="Print a single field from the resolved config")
    parser.add_argument(
        "--cli-args",
        action="store_true",
        help="Print shell-escaped CLI args for train.py",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.field:
        value = config[args.field]
        if isinstance(value, (dict, list)):
            print(json.dumps(value))
        else:
            print(value)
        return

    if args.cli_args:
        skip = {
            "model",
            "data_path",
            "test_data_path",
            "dinov3_repo_path",
            "dinov3_weights_path",
        }
        print(to_cli_args(config, skip))
        return

    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
