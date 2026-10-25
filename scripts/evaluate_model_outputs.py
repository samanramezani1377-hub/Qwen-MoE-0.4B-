#!/usr/bin/env python3
"""Compare dense Qwen outputs with a layer-0 top-k MoE replacement.

This is an output-level *partial-conversion* evaluation, not a complete MoE model
benchmark. The same base model is used for both passes to avoid loading two 2B
models into RAM. It reports next-token logit divergence, held-out text perplexity,
and greedy continuation differences.
"""
from __future__ import annotations
import argparse, difflib, json, math, time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill_real_qwen_ffn import MODEL_ID, get_text_model, make_moe_from_ffn

START = time.monotonic()

def log(event, **fields):
    print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic()-START, 2), **fields}, ensure_ascii=False), flush=True)

def get_logits(model, batch):
    with torch.inference_mode():
        return model(**batch, use_cache=False).logits.float()

def loss_sum(logits, input_ids, attention_mask):
    # Next-token prediction loss, ignoring padded positions.
    labels = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    values = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return float(values[valid].sum()), int(valid.sum())

def token_edit_metrics(a, b):
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ratio = matcher.ratio()
    # SequenceMatcher's ratio is descriptive similarity, not a semantic-quality score.
    return {
        "dense_generated_tokens": len(a),
        "moe_generated_tokens": len(b),
        "exact_token_match": a == b,
        "sequence_matcher_similarity_pct": round(100 * ratio, 3),
        "token_difference_pct_relative_to_dense_length": round(
            100 * (1 - ratio), 3
        ),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-text-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--max-documents", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no paid GPU is provisioned.")
    device = torch.device(a.device)
    docs = [s.strip() for s in a.eval_text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(docs) < 3:
        raise SystemExit(f"Need at least 3 independent held-out documents, got {len(docs)}")
    docs = docs[:a.max_documents]
    log("loading_model", model=a.model, device=str(device), documents=len(docs))
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if device.type == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    text_model = get_text_model(model)
    if not 0 <= a.layer < len(text_model.layers):
        raise ValueError(f"Invalid layer {a.layer}; model has {len(text_model.layers)} layers")
    layer = text_model.layers[a.layer]
    dense_ffn = layer.mlp
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    moe = make_moe_from_ffn(dense_ffn, a.num_experts, a.top_k).to(device)
    missing, unexpected = moe.load_state_dict(checkpoint["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    moe.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    dense_loss_sum = moe_loss_sum = 0.0
    target_tokens = 0
    abs_sum = sq_sum = dense_sq_sum = kl_sum = 0.0
    top1_matches = top5_matches = compared_positions = 0
    generation_results = []
    for index, doc in enumerate(docs, 1):
        encoded = tokenizer(
            doc, return_tensors="pt", truncation=True, max_length=a.max_length
        )
        batch = {k: v.to(device) for k, v in encoded.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if ids.shape[1] < 3:
            continue

        # Dense teacher pass and its greedy continuation.
        dense_logits = get_logits(model, batch)
        ds, count = loss_sum(dense_logits, ids, mask)
        dense_loss_sum += ds
        target_tokens += count
        with torch.inference_mode():
            dense_generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False,
                max_new_tokens=a.max_new_tokens, use_cache=True,
                pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id
            )
        dense_new = dense_generated[0, ids.shape[1]:].detach().cpu().tolist()
        dense_text = tokenizer.decode(dense_new, skip_special_tokens=True)

        # Replace only the selected FFN, evaluate student pass, then restore it.
        layer.mlp = moe
        moe_logits = get_logits(model, batch)
        ms, _ = loss_sum(moe_logits, ids, mask)
        moe_loss_sum += ms
        t = dense_logits[:, :-1, :].float()
        s = moe_logits[:, :-1, :].float()
        valid = mask[:, 1:].bool()
        tv, sv = t[valid], s[valid]
        abs_sum += float((tv - sv).abs().sum())
        sq_sum += float(((tv - sv) ** 2).sum())
        dense_sq_sum += float((tv ** 2).sum())
        logp = F.log_softmax(tv, dim=-1)
        logq = F.log_softmax(sv, dim=-1)
        kl_sum += float((logp.exp() * (logp - logq)).sum())
        dense_top1 = tv.argmax(dim=-1)
        moe_top1 = sv.argmax(dim=-1)
        top1_matches += int((dense_top1 == moe_top1).sum())
        top5 = tv.topk(min(5, tv.shape[-1]), dim=-1).indices
        student_top1 = moe_top1.unsqueeze(-1)
        top5_matches += int((top5 == student_top1).any(dim=-1).sum())
        compared_positions += int(tv.shape[0])
        del dense_logits, moe_logits, t, s, tv, sv

        with torch.inference_mode():
            moe_generated = model.generate(
                input_ids=ids, attention_mask=mask, do_sample=False,
                max_new_tokens=a.max_new_tokens, use_cache=True,
                pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id
            )
        moe_new = moe_generated[0, ids.shape[1]:].detach().cpu().tolist()
        moe_text = tokenizer.decode(moe_new, skip_special_tokens=True)
        layer.mlp = dense_ffn
        generation_results.append({
            "document_index": index,
            "prompt_prefix": doc[:240],
            "dense_continuation": dense_text,
            "moe_continuation": moe_text,
            "comparison": token_edit_metrics(dense_new, moe_new),
        })
        log("document_evaluated", document=index, total_documents=len(docs),
            token_positions_compared=compared_positions)

    if target_tokens == 0 or compared_positions == 0:
        raise RuntimeError("No valid token positions were evaluated")
    dense_ce = dense_loss_sum / target_tokens
    moe_ce = moe_loss_sum / target_tokens
    dense_ppl, moe_ppl = math.exp(min(dense_ce, 80)), math.exp(min(moe_ce, 80))
    report = {
        "status": "ok",
        "model": a.model,
        "device": str(device),
        "conversion_scope": "Only layer-0 (or selected layer) FFN is replaced by the trained MoE; all other layers remain dense.",
        "architecture": {"layer": a.layer, "num_experts": a.num_experts, "top_k": a.top_k},
        "evaluation": {
            "documents": len(generation_results),
            "target_tokens_for_cross_entropy": target_tokens,
            "logit_positions_compared": compared_positions,
            "max_length": a.max_length,
            "max_new_tokens": a.max_new_tokens,
            "held_out_document_file": str(a.eval_text_file),
        },
        "next_token_logits": {
            "mean_absolute_logit_error": abs_sum / max(1, compared_positions * int(getattr(text_model.config, "vocab_size", 1))),
            "relative_l2_error_pct": 100 * math.sqrt(sq_sum) / max(math.sqrt(dense_sq_sum), 1e-30),
            "top1_token_agreement_pct": 100 * top1_matches / compared_positions,
            "dense_top5_contains_moe_top1_pct": 100 * top5_matches / compared_positions,
            "mean_kl_dense_to_moe_nats_per_position": kl_sum / compared_positions,
            "interpretation": "Logit-space divergence; not itself a percentage of answer quality.",
        },
        "held_out_text_likelihood": {
            "dense_cross_entropy_nats_per_token": dense_ce,
            "moe_cross_entropy_nats_per_token": moe_ce,
            "dense_perplexity": dense_ppl,
            "moe_perplexity": moe_ppl,
            "cross_entropy_increase_pct": 100 * (moe_ce / dense_ce - 1) if dense_ce else None,
            "perplexity_increase_pct": 100 * (moe_ppl / dense_ppl - 1) if dense_ppl else None,
            "interpretation": "Relative deterioration on this held-out corpus only; not a universal human-rated quality-loss percentage.",
        },
        "greedy_continuation_comparison": generation_results,
        "limitations": [
            "This is not a complete all-layer MoE model; only one FFN is replaced.",
            "Generated-token difference or text similarity is not equivalent to semantic quality loss.",
            "Small Persian corpus results are screening evidence, not a general reasoning benchmark.",
            "No Android-device speed or memory result is measured here.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("evaluation_complete", report=str(a.output), documents=len(generation_results),
        relative_logit_l2_error_pct=report["next_token_logits"]["relative_l2_error_pct"],
        perplexity_increase_pct=report["held_out_text_likelihood"]["perplexity_increase_pct"],
        top1_agreement_pct=report["next_token_logits"]["top1_token_agreement_pct"])
if __name__ == "__main__":
    main()
