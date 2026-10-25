#!/usr/bin/env python3
"""Validate a teacher JSONL archive and export deterministic, non-destructive train/eval subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path, repair_trailing_line: bool) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    invalid_tail_bytes = b""
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("JSONL record is not an object")
            if not row.get("sample_id") or not (row.get("response_text") or row.get("text")):
                raise ValueError("Record is missing sample_id or response text")
            rows.append(row)
        except Exception as exc:
            is_last_nonempty = not any(rest.strip() for rest in lines[index + 1 :])
            if repair_trailing_line and is_last_nonempty:
                invalid_tail_bytes = b"".join(lines[index:])
                break
            raise RuntimeError(
                f"Invalid JSONL record at line {index + 1}; archive was not modified. "
                f"Error: {exc}"
            ) from exc
    if invalid_tail_bytes:
        quarantine = path.with_suffix(path.suffix + ".incomplete_tail")
        with quarantine.open("ab") as stream:
            stream.write(invalid_tail_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        valid_prefix = b"".join(lines[: len(lines) - len(invalid_tail_bytes.splitlines(keepends=True))])
        temp = path.with_suffix(path.suffix + ".repair.tmp")
        temp.write_bytes(valid_prefix)
        os.replace(temp, path)
    # Deduplicate by sample_id, preserving first-seen order; never dedupe away distinct prompts.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row["sample_id"])
        if sid not in seen:
            seen.add(sid)
            deduped.append(row)
    return deduped, len(invalid_tail_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Persistent full teacher JSONL archive")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, required=True, help="Selected train responses, one per line")
    parser.add_argument("--eval-text", type=Path, required=True, help="Held-out eval responses, one per line")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--repair-trailing-line", action="store_true",
                        help="Quarantine and remove only an invalid final/incomplete JSONL record")
    args = parser.parse_args()
    if not 0.05 <= args.eval_fraction <= 0.5:
        parser.error("--eval-fraction must be between 0.05 and 0.5")
    if args.max_train_samples < 1:
        parser.error("--max-train-samples must be >= 1")

    rows, quarantined_bytes = read_archive(args.archive, args.repair_trailing_line)
    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 valid teacher records; found {len(rows)}.")

    # Stable split independent of row ordering or future additions to the archive.
    train_pool: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    duplicates = 0
    for row in rows:
        prompt = str(row.get("prompt_text_as_tokenized") or row.get("prompt_source_text") or row.get("prompt") or "")
        response = str(row.get("response_text") or row.get("text") or "")
        pair_hash = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
        if pair_hash in seen_pairs:
            duplicates += 1
            continue
        seen_pairs.add(pair_hash)
        split_hash = hashlib.sha256(("teacher-archive-split-v1:" + str(row["sample_id"])).encode()).hexdigest()
        bucket = int(split_hash[:8], 16) / 0xFFFFFFFF
        (eval_rows if bucket < args.eval_fraction else train_pool).append(row)

    if not eval_rows and len(rows) >= 2:
        # Deterministic fallback ensures a held-out set exists.
        eval_rows.append(train_pool.pop())
    train_rows = train_pool[: args.max_train_samples]

    def write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def response(row: dict[str, Any]) -> str:
        return " ".join(str(row.get("response_text") or row.get("text") or "").split())

    def write_text(path: Path, data: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(t for t in (response(row) for row in data) if t) + "\n"
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)

    write_jsonl(args.train_jsonl, train_rows)
    write_jsonl(args.eval_jsonl, eval_rows)
    write_text(args.train_text, train_rows)
    write_text(args.eval_text, eval_rows)

    total_response_tokens = sum(
        len(row.get("response_token_ids", []))
        if isinstance(row.get("response_token_ids"), list)
        else 0
        for row in rows
    )
    manifest = {
        "schema_version": 1,
        "archive_path": str(args.archive),
        "archive_sha256": sha256_file(args.archive),
        "archive_records": len(rows),
        "archive_response_tokens_from_saved_token_ids": total_response_tokens,
        "duplicate_sample_ids_removed_from_views": len(rows) - len({str(r["sample_id"]) for r in rows}),
        "duplicate_prompt_response_pairs_excluded_from_views": duplicates,
        "eval_fraction_target": args.eval_fraction,
        "eval_records": len(eval_rows),
        "train_pool_records_before_cap": len(train_pool),
        "train_records_selected": len(train_rows),
        "train_sample_cap": args.max_train_samples,
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "train_response_text": str(args.train_text),
        "eval_response_text": str(args.eval_text),
        "quarantined_incomplete_tail_bytes": quarantined_bytes,
        "selection_policy": "Stable sample_id hash split; archive remains the source of truth; only a capped train subset is exported.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
