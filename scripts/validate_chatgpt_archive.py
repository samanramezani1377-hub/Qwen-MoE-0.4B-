#!/usr/bin/env python3
"""Validate the ChatGPT prompt/response JSONL archive used by hybrid KD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chatgpt_prompt_response_archive_part2.jsonl"),
        help="Path to the JSONL archive",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Archive not found: {args.input}")

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    count = 0
    duplicates = 0
    with args.input.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise SystemExit(f"Blank line at {line_no}; JSONL must have one record per line")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Line {line_no} must be a JSON object")

            sample_id = row.get("sample_id")
            prompt = row.get("prompt")
            answer = row.get("chatgpt_response")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise SystemExit(f"Line {line_no}: sample_id must be a non-empty string")
            if not isinstance(prompt, str) or not prompt.strip():
                raise SystemExit(f"Line {line_no}: prompt must be a non-empty string")
            if not isinstance(answer, str) or not answer.strip():
                raise SystemExit(f"Line {line_no}: chatgpt_response must be a non-empty string")
            if sample_id in seen_ids:
                raise SystemExit(f"Duplicate sample_id {sample_id!r} at line {line_no}")
            seen_ids.add(sample_id)
            pair = (prompt.strip(), answer.strip())
            if pair in seen_pairs:
                duplicates += 1
            seen_pairs.add(pair)
            count += 1

    if count == 0:
        raise SystemExit("Archive contains no records")
    print(json.dumps({
        "status": "success",
        "file": str(args.input),
        "records": count,
        "unique_sample_ids": len(seen_ids),
        "duplicate_prompt_response_pairs": duplicates,
    }, ensure_ascii=False, indent=2))
    if duplicates:
        raise SystemExit("Duplicate prompt/response pairs found; review before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
