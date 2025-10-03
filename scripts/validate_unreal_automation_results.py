#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


FAILURE_PATTERNS = (
    "Automation Test Failed",
    "LogAutomationController: Error:",
    "LogAutomation: Error:",
    "Fatal error",
    "Assertion failed",
    "Ensure condition failed",
)


def fail(message):
    print(f"[unreal-results] {message}", file=sys.stderr)
    return 1


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def contains_failure(text):
    text_lower = text.lower()
    for pattern in FAILURE_PATTERNS:
        if pattern.lower() in text_lower:
            return pattern
    return None


def evidence_token(test_filter):
    token = test_filter.split("*", 1)[0].strip()
    if token.endswith("."):
        token = token[:-1]
    return token or "AstralRT"


def main():
    parser = argparse.ArgumentParser(
        description="Validate Unreal Automation logs and report artifacts."
    )
    parser.add_argument("--log", required=True, help="Unreal Automation log file")
    parser.add_argument("--report-dir", required=True, help="Automation report directory")
    parser.add_argument("--filter", default="AstralRT.*", help="Automation test filter")
    args = parser.parse_args()

    log_path = Path(args.log)
    report_dir = Path(args.report_dir)
    token = evidence_token(args.filter)

    if not log_path.is_file() or log_path.stat().st_size == 0:
        return fail(f"missing or empty Unreal log: {log_path}")
    if not report_dir.is_dir():
        return fail(f"missing Automation report directory: {report_dir}")

    report_files = [path for path in report_dir.rglob("*") if path.is_file() and path.stat().st_size > 0]
    if not report_files:
        return fail(f"Automation report directory has no non-empty files: {report_dir}")

    combined = read_text(log_path)
    for path in report_files:
        combined += "\n"
        combined += read_text(path)

    failure = contains_failure(combined)
    if failure:
        return fail(f"Automation output contains failure marker: {failure}")

    if token and token not in combined:
        return fail(f"Automation output does not mention expected test filter token: {token}")

    print(f"[unreal-results] OK: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
