#!/usr/bin/env python3
"""End-to-end held-out perplexity and next-token agreement for dense vs patched Qwen MoE.

Input file: one independent document per UTF-8 line. This evaluates an experimental runtime
built by replacing all text decoder MLPs with checkpoint-backed top-k MoE modules.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_runtime import patch_model_with_checkpoints


def get_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def evaluate(model, tokenizer, texts, device, max_length, teacher_top_ids=None):
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    compared = 0
    with torch.inference_mode():
        for i, text in enumerate(texts):
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).to(device)
            if ids.shape[1] < 2:
                continue
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :].float()
            labels = ids[:, 1:]
            valid = mask[:, 1:].bool()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            total_nll += float(nll[valid].sum().cpu())
            total_tokens += int(valid.sum().cpu())
            pred = logits.argmax(dim=-1)
            if teacher_top_ids is not None and i < len(teacher_top_ids):
                target = teacher_top_ids[i].to(pred.device)
                length = min(pred.shape[1], target.shape[1])
                m = valid[:, :length]
                correct += int(((pred[:, :length] == target[:, :length]) & m).sum().cpu())
                compared += int(m.sum().cpu())
    return {"tokens": total_tokens, "cross_entropy": total_nll/max(total_tokens,1),
            "perplexity": math.exp(min(total_nll/max(total_tokens,1), 30)),
            "next_token_agreement": (correct/compared if compared else None),
            "agreement_tokens": compared}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--text-file", type=Path, required=True, help="Independent held-out UTF-8 docs, one per line")
    p.add_argument("--output", type=Path, default=Path("artifacts/full_model_eval.json"))
    p.add_argument("--max-documents", type=int, default=256)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    a = p.parse_args()
    if a.max_documents < 1 or a.max_length < 2:
        p.error("max-documents must be >=1 and max-length >=2")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    texts = [x.strip() for x in a.text_file.read_text(encoding="utf-8").splitlines() if x.strip()][:a.max_documents]
    if len(texts) < 2:
        raise ValueError("Provide at least two held-out documents; do not use training documents.")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    dtype = get_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    teacher_top_ids = []
    # Save only argmax token IDs, not full vocabulary logits, to keep memory bounded.
    with torch.inference_mode():
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=a.max_length)
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).to(device)
            if ids.shape[1] < 2:
                teacher_top_ids.append(torch.empty((1,0), dtype=torch.long))
                continue
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :].float()
            teacher_top_ids.append(logits.argmax(dim=-1).cpu())
    teacher = evaluate(model, tokenizer, texts, device, a.max_length)
    patch_report = patch_model_with_checkpoints(model, a.checkpoint_dir, expected_layers=24,
                                                num_experts=a.num_experts, top_k=a.top_k,
                                                map_location="cpu")
    converted = evaluate(model, tokenizer, texts, device, a.max_length, teacher_top_ids)
    report = {"status": "evaluation_completed", "model_id": a.model, "device": str(device),
              "checkpoint_dir": str(a.checkpoint_dir), "documents": len(texts),
              "max_length": a.max_length, "num_experts": a.num_experts, "top_k": a.top_k,
              "teacher": teacher, "converted": converted,
              "perplexity_ratio_converted_over_teacher":
                  converted["perplexity"]/teacher["perplexity"] if teacher["perplexity"] else None,
              "cross_entropy_delta": converted["cross_entropy"]-teacher["cross_entropy"],
              "patch_report": patch_report,
              "scope_note": "End-to-end language-model evaluation on supplied documents. This does not test reasoning comprehensively or Android latency. Run only with an independent held-out corpus."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
