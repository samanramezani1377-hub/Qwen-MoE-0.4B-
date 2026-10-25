#!/usr/bin/env python3
"""Capture Qwen next-token logits aligned to ChatGPT-written responses for offline KD."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} must be a JSON object")
            sample_id = str(row.get("sample_id", "")).strip()
            prompt = str(row.get("prompt") or row.get("prompt_text") or "").strip()
            answer = str(row.get("chatgpt_response") or row.get("response_text") or row.get("answer") or "").strip()
            if not sample_id or not prompt or not answer:
                raise ValueError(
                    f"Line {line_no} needs sample_id, prompt (or prompt_text), "
                    "and chatgpt_response (or response_text/answer)"
                )
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id {sample_id!r} on line {line_no}")
            seen.add(sample_id)
            row["_sample_id"], row["_prompt"], row["_answer"] = sample_id, prompt, answer
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True,
                        help="ChatGPT prompt/answer JSONL; one object per sample")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B",
                        help="Exact Qwen teacher model ID or local model directory")
    parser.add_argument("--revision", default=None, help="Optional pinned Hugging Face revision")
    parser.add_argument("--top-k", type=int, default=64,
                        help="Store top-k logits per answer position (sparse approximation)")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0, help="KD temperature used for top-k logits and full-vocabulary log-normalizer")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 means all samples; use a small value for smoke tests")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.top_k < 1 or args.max_seq_length < 16 or args.temperature <= 0:
        parser.error("--top-k must be >=1, --max-seq-length >=16, and --temperature >0")
    rows = read_jsonl(args.input_jsonl)
    if args.max_samples:
        rows = rows[:args.max_samples]
    if not rows:
        raise RuntimeError("No valid samples found in input JSONL")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = args.output_dir / "qwen_topk_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "Tokenizer has no chat_template. Do not guess a chat format; use the model's "
            "documented template or add an explicit template before running."
        )
    model_kwargs: dict[str, Any] = {"revision": args.revision, "trust_remote_code": False}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    model_device = next(model.parameters()).device
    model_vocab = int(model.config.vocab_size)
    if args.top_k > model_vocab:
        raise ValueError(f"--top-k={args.top_k} exceeds model vocabulary size {model_vocab}")
    actual_k = min(args.top_k, model_vocab)
    manifest_path = args.output_dir / "qwen_logits_manifest.json"
    index_path = args.output_dir / "qwen_logits_index.jsonl"
    completed = 0
    skipped = 0
    # Existing outputs from the earlier extractor may contain top-k values but lack
    # input_ids/log-normalizers. Reuse only complete, compatible tensors already indexed.
    indexed_ids: set[str] = set()
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as prior_index:
            for prior_line in prior_index:
                if not prior_line.strip():
                    continue
                try:
                    prior_row = json.loads(prior_line)
                    if prior_row.get("sample_id"):
                        indexed_ids.add(str(prior_row["sample_id"]))
                except json.JSONDecodeError:
                    # An interrupted final index line is ignored; the tensor will be regenerated/indexed.
                    continue

    with index_path.open("a", encoding="utf-8") as index:
        for row in rows:
            sid = row["_sample_id"]
            safe_id = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:24]
            tensor_path = logits_dir / f"{safe_id}.pt"
            if tensor_path.exists() and sid in indexed_ids:
                try:
                    existing = torch.load(tensor_path, map_location="cpu", weights_only=True)
                    required_keys = {"input_ids", "target_token_ids", "topk_token_ids",
                                     "topk_logits", "teacher_logsumexp", "temperature"}
                    compatible = required_keys.issubset(existing) and abs(
                        float(existing["temperature"]) - args.temperature
                    ) < 1e-6
                except Exception:
                    compatible = False
                if compatible:
                    skipped += 1
                    continue
            prefix_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["_prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            if isinstance(prefix_ids, dict):
                prefix_ids = prefix_ids["input_ids"]
            if prefix_ids and isinstance(prefix_ids[0], list):
                prefix_ids = prefix_ids[0]
            answer_ids = tokenizer.encode(row["_answer"], add_special_tokens=False)
            if not answer_ids:
                raise RuntimeError(f"Answer tokenized to empty sequence for sample {sid}")
            # Need one prior token to predict the first answer token.
            budget = args.max_seq_length - len(prefix_ids)
            if budget <= 0:
                prefix_ids = prefix_ids[-(args.max_seq_length - 1):]
                budget = args.max_seq_length - len(prefix_ids)
            answer_ids = answer_ids[:budget]
            if not answer_ids:
                raise RuntimeError(f"No answer tokens fit max sequence length for sample {sid}")
            all_ids = prefix_ids + answer_ids
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=model_device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids)
                # Position i predicts token i+1. Align to answer token positions only.
                start = len(prefix_ids) - 1
                end = start + len(answer_ids)
                answer_logits = outputs.logits[0, start:end, :]  # Only the answer positions are retained
                scaled_logits = answer_logits.float() / args.temperature
                teacher_logsumexp = torch.logsumexp(scaled_logits, dim=-1)
                values, indices = torch.topk(scaled_logits, k=actual_k, dim=-1)
                # Store temperature-scaled top-k logits plus the full-vocabulary log-normalizer.
                # The trainer uses both to recover top-k probabilities and aggregated tail mass.
                payload = {
                    "topk_token_ids": indices.to(device="cpu", dtype=torch.int32),
                    "topk_logits": values.to(device="cpu", dtype=torch.float32),
                    "teacher_logsumexp": teacher_logsumexp.to(device="cpu", dtype=torch.float32),
                    "input_ids": torch.tensor(all_ids, dtype=torch.int32),
                    "target_token_ids": torch.tensor(answer_ids, dtype=torch.int32),
                    "prefix_token_count": len(prefix_ids),
                    "answer_token_count": len(answer_ids),
                    "vocab_size": model_vocab,
                    "top_k": actual_k,
                    "temperature": args.temperature,
                }
                torch.save(payload, tensor_path)
            index_row = {
                "sample_id": sid,
                "prompt": row["_prompt"],
                "chatgpt_response": row["_answer"],
                "logits_file": str(tensor_path.relative_to(args.output_dir)),
                "prompt_tokens": len(prefix_ids),
                "answer_tokens_captured": len(answer_ids),
                "answer_tokens_original": len(tokenizer.encode(row["_answer"], add_special_tokens=False)),
                "model_id": args.model,
                "model_revision": args.revision,
                "storage_format": "top-k sparse logits; not a full-vocabulary logit tensor",
            }
            index.write(json.dumps(index_row, ensure_ascii=False) + "\n")
            index.flush()
            completed += 1
            if completed % 10 == 0:
                print(f"Captured {completed} new samples; skipped existing files: {skipped}", flush=True)

    manifest = {
        "status": "success",
        "model_id": args.model,
        "model_revision": args.revision,
        "input_jsonl": str(args.input_jsonl),
        "input_samples_selected": len(rows),
        "new_samples_captured": completed,
        "existing_samples_skipped": skipped,
        "top_k": actual_k,
        "vocab_size": model_vocab,
        "max_seq_length": args.max_seq_length,
        "temperature": args.temperature,
        "tokenizer_id": args.model,
        "storage_format": "sparse top-k logits per answer token; approximate KD signal",
        "important_limit": "Top-k logits omit individual tail classes; teacher_logsumexp preserves the full-vocabulary normalization so the trainer can aggregate the omitted tail as one bucket. This is an approximation, not exact full-vocabulary KL. Use a student with the same tokenizer/vocabulary.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
