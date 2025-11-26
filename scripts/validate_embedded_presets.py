#!/usr/bin/env python3
"""Validate embedded CMake presets keep filesystem-heavy features disabled."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EMBEDDED_PRESETS = {
    "embedded-x86_64",
    "embedded-x86_64-noexceptions",
    "embedded-native",
    "embedded-arm64-cross",
    "embedded-arm64-ci",
    "embedded-armv7-cross",
    "embedded-armv7-ci",
}

REQUIRED_OFF = (
    "ASTRAL_ENABLE_VIRTUAL_MEMORY",
    "ASTRAL_CPU_MEMORY_SOURCE_MMAP",
    "ASTRAL_ENABLE_JSON_SCHEMA_GRAMMAR",
    "ASTRAL_ENABLE_DYNAMIC_BACKENDS",
    "ASTRAL_ENABLE_THREADS",
)


def fail(message: str) -> int:
    print(f"[embedded-presets] {message}", file=sys.stderr)
    return 1


def normalize_bool(value: object) -> str:
    if isinstance(value, str):
        return value.upper()
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value).upper()


def preset_chain(name: str, presets: dict[str, dict], visiting: set[str] | None = None) -> list[dict]:
    visiting = set() if visiting is None else visiting
    if name in visiting:
        raise ValueError(f"cyclic preset inheritance at {name}")
    if name not in presets:
        raise ValueError(f"missing preset {name}")

    visiting.add(name)
    preset = presets[name]
    inherits = preset.get("inherits", [])
    if isinstance(inherits, str):
        inherits = [inherits]
    if not isinstance(inherits, list):
        raise ValueError(f"{name}.inherits must be a string or list")

    chain: list[dict] = []
    for parent in inherits:
        if not isinstance(parent, str):
            raise ValueError(f"{name}.inherits entries must be strings")
        chain.extend(preset_chain(parent, presets, visiting))
    chain.append(preset)
    visiting.remove(name)
    return chain


def resolved_cache(name: str, presets: dict[str, dict]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for preset in preset_chain(name, presets):
        cache = preset.get("cacheVariables", {})
        if not isinstance(cache, dict):
            raise ValueError(f"{preset.get('name', name)}.cacheVariables must be an object")
        merged.update(cache)
    return merged


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("CMakePresets.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        configure_presets = data.get("configurePresets")
        if not isinstance(configure_presets, list):
            return fail("CMakePresets.json must contain configurePresets")
        presets = {
            preset["name"]: preset
            for preset in configure_presets
            if isinstance(preset, dict) and isinstance(preset.get("name"), str)
        }
        missing = sorted(EMBEDDED_PRESETS - set(presets))
        if missing:
            return fail(f"missing embedded preset(s): {', '.join(missing)}")

        for name in sorted(EMBEDDED_PRESETS):
            cache = resolved_cache(name, presets)
            for key in REQUIRED_OFF:
                actual = normalize_bool(cache.get(key))
                if actual != "OFF":
                    return fail(f"{name} must set {key}=OFF after inheritance, got {actual}")
    except OSError as exc:
        return fail(str(exc))
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        return fail(str(exc))

    print(f"[embedded-presets] OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
