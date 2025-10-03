#!/usr/bin/env python3
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def fail(message):
    print(f"[unity-results] {message}", file=sys.stderr)
    return 1


def int_attr(node, name):
    value = node.attrib.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{node.tag}.{name} is not an integer: {value}") from exc


def main():
    parser = argparse.ArgumentParser(description="Validate Unity EditMode test result XML.")
    parser.add_argument("results", help="Unity EditMode result XML")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.is_file() or path.stat().st_size == 0:
        return fail(f"missing or empty Unity result XML: {path}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return fail(f"malformed Unity result XML: {exc}")
    except OSError as exc:
        return fail(str(exc))

    root = tree.getroot()
    failed = int_attr(root, "failed")
    if failed is not None and failed != 0:
        return fail(f"Unity result reports failed={failed}")

    result = root.attrib.get("result")
    if result is not None and result.lower() not in ("passed", "success"):
        return fail(f"Unity result is not passed: {result}")

    test_cases = list(root.iter("test-case"))
    if not test_cases:
        return fail("Unity result XML does not contain any test-case entries")

    for case in test_cases:
        case_name = case.attrib.get("fullname") or case.attrib.get("name") or "<unnamed>"
        case_result = case.attrib.get("result", "")
        if case_result.lower() not in ("passed", "success"):
            return fail(f"Unity test did not pass: {case_name} ({case_result})")

    print(f"[unity-results] OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
