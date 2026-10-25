#!/usr/bin/env python3
"""Train the existing full Qwen MoE overlay with ChatGPT targets + Qwen sparse-logit KD.

This is the single intended training path. It does not offer text-only or logits-only
training modes. Teacher logits must be captured first by capture_qwen_logits_for_chatgpt_data.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM
from transformers.optimization import Adafactor

from moe_runtime import get_text_model, patch_model_with_checkpoints


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "time": round(time.time(), 3), **fields},
                     ensure_ascii=False), flush=True)


def load_index(index_path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    """Read a JSONL index, deduplicate by sample_id, and resolve each tensor file."""
    by_id: dict[str, dict[str, Any]] = {}
    with index_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid index JSON on line {line_no}: {exc}") from exc
            sid = str(row.get("sample_id", "")).strip()
            rel = row.get("logits_file")
            if not sid or not rel:
                raise ValueError(f"Index line {line_no} needs sample_id and logits_file")
            # The last valid row wins if a previous interrupted run appended a duplicate.
            row["_tensor_path"] = (index_path.parent / rel).resolve()
            by_id[sid] = row
    rows = list(by_id.values())
    missing = [str(r["_tensor_path"]) for r in rows if not r["_tensor_path"].is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} indexed tensor files are missing; first examples: {missing[:3]}. "
            "Re-run the logits extractor to repair the index before training."
        )
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError(f"No indexed samples with tensor files found beside {index_path}")
    return rows


def load_sample(row: dict[str, Any], expected_vocab: int | None,
                temperature: float) -> dict[str, torch.Tensor | int | str]:
    payload = torch.load(row["_tensor_path"], map_location="cpu", weights_only=True)
    required = ("input_ids", "target_token_ids", "topk_token_ids", "topk_logits",
                "teacher_logsumexp", "prefix_token_count", "answer_token_count", "vocab_size")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"{row['_tensor_path']} is missing {missing}. Re-run the updated logits extractor; "
            "legacy top-k files do not contain the full sequence and tail-mass metadata."
        )
    vocab = int(payload["vocab_size"])
    if expected_vocab is not None and vocab != expected_vocab:
        raise ValueError(f"Vocabulary mismatch in {row['_tensor_path']}: {vocab} vs {expected_vocab}")
    saved_temp = float(payload.get("temperature", temperature))
    if not math.isclose(saved_temp, temperature, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            f"KD temperature mismatch in {row['_tensor_path']}: saved={saved_temp}, requested={temperature}"
        )
    input_ids = payload["input_ids"].long().reshape(-1)
    target = payload["target_token_ids"].long().reshape(-1)
    top_ids = payload["topk_token_ids"].long()
    top_logits = payload["topk_logits"].float()
    log_z = payload["teacher_logsumexp"].float().reshape(-1)
    prefix = int(payload["prefix_token_count"])
    answer_count = int(payload["answer_token_count"])
    if input_ids.numel() != prefix + answer_count:
        raise ValueError(f"Input/answer lengths disagree in {row['_tensor_path']}")
    if target.numel() != answer_count or top_ids.shape[0] != answer_count:
        raise ValueError(f"Target/top-k length mismatch in {row['_tensor_path']}")
    if top_logits.shape != top_ids.shape or log_z.numel() != answer_count:
        raise ValueError(f"Teacher-logit tensor shapes disagree in {row['_tensor_path']}")
    start = prefix - 1
    if start < 0 or start + answer_count > input_ids.numel() - 1:
        raise ValueError(f"Invalid prefix boundary in {row['_tensor_path']}")
    if not torch.equal(input_ids[prefix:prefix + answer_count], target):
        raise ValueError(f"Saved target tokens do not match input sequence in {row['_tensor_path']}")
    if top_ids.min().item() < 0 or top_ids.max().item() >= vocab:
        raise ValueError(f"Teacher top-k token ID outside vocabulary in {row['_tensor_path']}")
    return {
        "input_ids": input_ids,
        "target_token_ids": target,
        "topk_token_ids": top_ids,
        "topk_logits": top_logits,
        "teacher_logsumexp": log_z,
        "prefix_token_count": prefix,
        "answer_token_count": answer_count,
        "sample_id": str(row["sample_id"]),
    }


def sparse_aggregated_tail_kl(student_logits: torch.Tensor, sample: dict[str, Any],
                              temperature: float) -> torch.Tensor:
    """Approximate full-vocabulary KL using teacher top-k classes plus one aggregated tail.

    The teacher extractor stores logsumexp over the *entire* vocabulary at the same
    temperature. Teacher probability mass outside top-k and student mass outside those
    teacher-selected IDs are each aggregated into a single tail bucket. This is not
    exact full-vocabulary KL, but unlike renormalizing top-k alone it preserves tail mass.
    """
    top_ids = sample["topk_token_ids"].to(student_logits.device)
    teacher_top_logits = sample["topk_logits"].to(student_logits.device)
    teacher_log_z = sample["teacher_logsumexp"].to(student_logits.device)
    student_scaled = student_logits.float() / temperature
    student_log_probs = F.log_softmax(student_scaled, dim=-1)
    student_top_log_probs = student_log_probs.gather(-1, top_ids)
    teacher_top_log_probs = teacher_top_logits - teacher_log_z.unsqueeze(-1)
    teacher_top_probs = teacher_top_log_probs.exp().clamp_min(0.0)
    teacher_top_mass = teacher_top_probs.sum(dim=-1).clamp(max=1.0)
    teacher_tail = (1.0 - teacher_top_mass).clamp_min(1e-8)
    student_top_mass = student_top_log_probs.exp().sum(dim=-1).clamp(max=1.0)
    student_tail = (1.0 - student_top_mass).clamp_min(1e-8)
    top_kl = (
        teacher_top_probs
        * (teacher_top_log_probs - student_top_log_probs)
    ).sum(dim=-1)
    tail_kl = teacher_tail * (teacher_tail.log() - student_tail.log())
    return (top_kl + tail_kl).mean() * (temperature ** 2)


def checkpoint_payloads(model: torch.nn.Module, model_id: str, revision: str,
                        num_experts: int, top_k: int, step: int) -> Iterator[tuple[str, dict[str, Any]]]:
    # Yield one layer at a time so checkpointing does not duplicate all MoE weights in host RAM.
    text_model = get_text_model(model)
    for layer_idx, layer in enumerate(text_model.layers):
        mlp = layer.mlp
        state = {k: v.detach().to("cpu").contiguous() for k, v in mlp.state_dict().items()}
        yield (
            f"layer_{layer_idx}_top{top_k}_{num_experts}experts.pt",
            {
                "layer": layer_idx,
                "num_experts": num_experts,
                "top_k": top_k,
                "model_id": model_id,
                "teacher_revision": revision,
                "state_dict": state,
                "training_method": "chatgpt_text_plus_qwen_sparse_logits_kd",
                "training_step": step,
            },
        )


def save_state(model: torch.nn.Module, output_dir: Path, model_id: str, revision: str,
               num_experts: int, top_k: int, step: int, optimizer: Any,
               scaler: Any, temperature: float, kd_weight: float) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in checkpoint_payloads(model, model_id, revision, num_experts, top_k, step):
        path = checkpoint_dir / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
    state = {
        "step": step,
        "model_id": model_id,
        "teacher_revision": revision,
        "num_experts": num_experts,
        "top_k": top_k,
        "kd_temperature": temperature,
        "kd_weight": kd_weight,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    tmp_state = output_dir / "training_state.pt.tmp"
    torch.save(state, tmp_state)
    tmp_state.replace(output_dir / "training_state.pt")
    emit("checkpoint_saved", step=step, checkpoint_dir=str(checkpoint_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-jsonl", type=Path, required=True,
                        help="qwen_logits_index.jsonl produced by the updated extractor")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing the 24 initial MoE FFN checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default=None,
                        help="Must match checkpoint teacher_revision; inferred when omitted")
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every indexed sample")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Must match extractor temperature; default 1.0")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="Resume optimizer/step from output-dir/training_state.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_samples < 0 or args.max_steps < 0:
        parser.error("epochs must be >=1 and sample/step limits nonnegative")
    if args.learning_rate <= 0 or args.kd_weight < 0 or args.temperature <= 0:
        parser.error("learning-rate/temperature must be positive and kd-weight nonnegative")
    if args.save_every < 1 or args.max_grad_norm <= 0:
        parser.error("save-every and max-grad-norm must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    initial_dir = args.output_dir / "checkpoints" if args.resume else args.checkpoint_dir
    first_files = sorted(initial_dir.glob(f"layer_*_top{args.top_k}_{args.num_experts}experts.pt"))
    if not first_files:
        raise FileNotFoundError(f"No compatible MoE checkpoints in {initial_dir}")
    first_payload = torch.load(first_files[0], map_location="cpu", weights_only=False)
    revision = args.revision or first_payload.get("teacher_revision")
    if not revision:
        raise ValueError("No immutable teacher_revision in initial checkpoints; refusing an unpinned model load")
    if first_payload.get("model_id") and first_payload["model_id"] != args.model:
        raise ValueError(f"Checkpoint model_id={first_payload['model_id']!r} does not match --model={args.model!r}")
    rows = load_index(args.index_jsonl, args.max_samples)
    random.Random(args.seed).shuffle(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    emit("training_start", samples=len(rows), epochs=args.epochs, model=args.model,
         revision=revision, device=str(device), dtype=str(dtype), kd_weight=args.kd_weight,
         temperature=args.temperature, method="chatgpt_text_plus_qwen_sparse_logits_kd")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    patch_report = patch_model_with_checkpoints(
        model, initial_dir, expected_layers=24, num_experts=args.num_experts,
        top_k=args.top_k, map_location="cpu",
    )
    # Freeze the dense backbone; train only the installed MoE FFNs and their routers.
    for param in model.parameters():
        param.requires_grad_(False)
    text_model = get_text_model(model)
    trainable = []
    for layer in text_model.layers:
        layer.mlp.requires_grad_(True)
        trainable.extend(p for p in layer.mlp.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("No trainable MoE parameters found after patching")
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()

    optimizer = Adafactor(
        trainable, lr=args.learning_rate, relative_step=False, scale_parameter=False,
        warmup_init=False, weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    start_step = 0
    if args.resume:
        state_path = args.output_dir / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but {state_path} does not exist")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("teacher_revision") != revision or state.get("model_id") != args.model:
            raise ValueError("Resume state base-model identity does not match current run")
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_step = int(state.get("step", 0))
        emit("resume_loaded", step=start_step)

    log_path = args.output_dir / "training_metrics.jsonl"
    step = start_step
    total_loss_sum = total_ce_sum = total_kd_sum = 0.0
    seen_steps = 0
    try:
        with log_path.open("a", encoding="utf-8") as metrics:
            first_epoch = start_step // len(rows)
            first_row = start_step % len(rows)
            for epoch in range(first_epoch, args.epochs):
                row_start = first_row if epoch == first_epoch else 0
                for row_index in range(row_start, len(rows)):
                    row = rows[row_index]
                    sample = load_sample(row, expected_vocab=int(model.config.vocab_size),
                                         temperature=args.temperature)
                    ids = sample["input_ids"].to(device).long().unsqueeze(0)
                    prefix = int(sample["prefix_token_count"])
                    answer_count = int(sample["answer_token_count"])
                    start = prefix - 1
                    targets = sample["target_token_ids"].to(device).long()
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda" and dtype == torch.float16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(input_ids=ids, use_cache=False)
                    else:
                        outputs = model(input_ids=ids, use_cache=False)
                    student_logits = outputs.logits[0, start:start + answer_count, :]
                    if student_logits.shape[0] != answer_count:
                        raise RuntimeError(f"Student answer logits misaligned for sample {sample['sample_id']}")
                    ce_loss = F.cross_entropy(student_logits.float(), targets)
                    kd_loss = sparse_aggregated_tail_kl(student_logits, sample, args.temperature)
                    loss = ce_loss + args.kd_weight * kd_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite combined loss at sample {sample['sample_id']}: "
                            f"CE={float(ce_loss.detach())}, KD={float(kd_loss.detach())}"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    step += 1
                    seen_steps += 1
                    total_loss_sum += float(loss.detach())
                    total_ce_sum += float(ce_loss.detach())
                    total_kd_sum += float(kd_loss.detach())
                    event = {
                        "event": "train_step", "step": step, "epoch": epoch + 1,
                        "sample_id": sample["sample_id"], "sequence_tokens": int(ids.shape[1]),
                        "answer_tokens": answer_count, "loss": float(loss.detach()),
                        "text_ce_loss": float(ce_loss.detach()), "qwen_kd_loss": float(kd_loss.detach()),
                        "grad_norm": float(grad_norm.detach().float().cpu()),
                        "learning_rate": args.learning_rate,
                    }
                    metrics.write(json.dumps(event, ensure_ascii=False) + "\n")
                    metrics.flush()
                    if step % 10 == 0:
                        emit("train_progress", step=step, epoch=epoch + 1,
                             loss=event["loss"], text_ce_loss=event["text_ce_loss"],
                             qwen_kd_loss=event["qwen_kd_loss"], grad_norm=event["grad_norm"])
                    if step % args.save_every == 0:
                        save_state(model, args.output_dir, args.model, revision,
                                   args.num_experts, args.top_k, step, optimizer, scaler,
                                   args.temperature, args.kd_weight)
                    if args.max_steps and step - start_step >= args.max_steps:
                        break
                if args.max_steps and step - start_step >= args.max_steps:
                    break
    except KeyboardInterrupt:
        emit("interrupted", step=step, note="Saving current MoE weights before exit")
        save_state(model, args.output_dir, args.model, revision, args.num_experts, args.top_k,
                   step, optimizer, scaler, args.temperature, args.kd_weight)
        raise
    save_state(model, args.output_dir, args.model, revision, args.num_experts, args.top_k,
               step, optimizer, scaler, args.temperature, args.kd_weight)
    summary = {
        "status": "training_completed",
        "method": "chatgpt_text_plus_qwen_sparse_logits_kd",
        "model_id": args.model,
        "teacher_revision": revision,
        "checkpoint_dir": str(args.output_dir / "checkpoints"),
        "samples_indexed": len(rows),
        "steps_this_run": seen_steps,
        "total_step": step,
        "mean_combined_loss": total_loss_sum / max(seen_steps, 1),
        "mean_text_ce_loss": total_ce_sum / max(seen_steps, 1),
        "mean_qwen_kd_loss": total_kd_sum / max(seen_steps, 1),
        "kd_weight": args.kd_weight,
        "temperature": args.temperature,
        "patch_report": patch_report,
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "note": "Training execution is not proof of quality; evaluate on a held-out set before deployment.",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    emit("training_completed", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
