#!/usr/bin/env python3
"""Train a top-2/32-expert replacement for every text-decoder FFN, one layer at a time.

This creates per-layer research checkpoints. It does NOT by itself create a loadable
full MoE Transformers model; use the companion packaging/runtime implementation
before claiming a complete model is ready.

Example:
  python scripts/distill_all_qwen_ffn_layers.py --text-file data/fa_train_documents.txt \
    --output-dir artifacts/full_moe --num-experts 32 --top-k 2 --epochs 3

Memory-conscious design: the dense teacher stays loaded, but only one layer's
activation pairs are retained in RAM at a time. Use a free Colab/Kaggle runtime
with enough RAM; no paid compute is provisioned by this script.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from distill_real_qwen_ffn import MODEL_ID, get_text_model, make_moe_from_ffn

START = time.monotonic()

def log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic()-START, 2), **fields},
                     ensure_ascii=False), flush=True)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--samples", type=int, default=0, help="0 means use all non-empty lines")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1, help="Documents per teacher forward pass")
    p.add_argument("--train-batch-size", type=int, default=256, help="Captured token vectors per MoE optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--balance-loss-weight", type=float, default=0.01)
    p.add_argument("--eval-fraction", type=float, default=0.1,
                   help="Fraction of whole documents held out for evaluation")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--end-layer", type=int, default=-1, help="-1 means last text-decoder layer")
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    if (a.epochs < 1 or a.batch_size < 1 or a.train_batch_size < 1
            or not 0.05 <= a.eval_fraction <= 0.5 or a.balance_loss_weight < 0):
        p.error("epochs/batch sizes must be positive, eval-fraction in [0.05,0.5], and balance-loss-weight nonnegative")
    if not 1 <= a.top_k <= a.num_experts:
        p.error("Require 1 <= top-k <= num-experts")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable; this script never provisions paid compute")
    device = torch.device("cuda" if a.device == "cuda" or
                          (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    texts = [s.strip() for s in a.text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    random.Random(a.seed).shuffle(texts)
    if a.samples > 0:
        texts = texts[:a.samples]
    if len(texts) < 4:
        p.error(f"Need at least 4 non-empty training samples, got {len(texts)}")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.bfloat16
    a.output_dir.mkdir(parents=True, exist_ok=True)
    log("loading_teacher", model=a.model, device=str(device), dtype=str(dtype), documents=len(texts))
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    teacher = AutoModel.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True,
                                        trust_remote_code=a.trust_remote_code).to(device).eval()
    text_model = get_text_model(teacher)
    layers = text_model.layers
    last_layer = len(layers) - 1 if a.end_layer < 0 else a.end_layer
    if not (0 <= a.start_layer <= last_layer < len(layers)):
        p.error(f"Invalid layer range {a.start_layer}..{last_layer}; model has {len(layers)} layers")
    # Freeze one corpus-level split for every layer. Otherwise a document held out
    # for layer 0 could leak into training for layer 1 and invalidate end-to-end eval.
    doc_perm = torch.randperm(len(texts), generator=torch.Generator().manual_seed(a.seed))
    n_eval_docs = max(1, int(a.eval_fraction * len(doc_perm)))
    eval_doc_ids = doc_perm[:n_eval_docs].tolist()
    train_doc_ids = doc_perm[n_eval_docs:].tolist()
    if not train_doc_ids:
        p.error("Document-level split produced no training documents")
    log("global_document_split_complete", train_documents=len(train_doc_ids),
        eval_documents=len(eval_doc_ids), total_documents=len(texts), seed=a.seed)
    for idx in range(a.start_layer, last_layer + 1):
        ffn = getattr(layers[idx], "mlp", None)
        if ffn is None or not all(hasattr(ffn, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise RuntimeError(f"Layer {idx} FFN is not a supported SwiGLU gate/up/down module")
        captures: list[tuple[torch.Tensor, torch.Tensor]] = []
        masks: list[torch.Tensor] = []
        def hook(_module, inputs, output):
            if inputs and inputs[0].ndim == 3 and output.ndim == 3:
                captures.append((inputs[0].detach().float().cpu(), output.detach().float().cpu()))
        handle = ffn.register_forward_hook(hook)
        log("layer_capture_started", layer=idx, total_layers=last_layer-a.start_layer+1)
        with torch.inference_mode():
            for start in range(0, len(texts), a.batch_size):
                batch_texts = texts[start:start+a.batch_size]
                batch = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=a.max_length)
                mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).bool()
                teacher(**{k: v.to(device) for k, v in batch.items()})
                masks.append(mask)
                log("layer_capture_progress", layer=idx,
                    documents_processed=min(start+a.batch_size, len(texts)), total_documents=len(texts))
        handle.remove()
        if len(captures) != len(masks) or not captures:
            raise RuntimeError(f"Layer {idx}: failed to capture all batches")
        # Keep each document intact so evaluation is held out by document, not token.
        x_docs, y_docs = [], []
        for (xb, yb), mask in zip(captures, masks):
            for row in range(xb.shape[0]):
                valid = mask[row]
                if valid.any():
                    x_docs.append(xb[row][valid])
                    y_docs.append(yb[row][valid])
        del captures, masks
        x_train, y_train = torch.cat([x_docs[i] for i in train_doc_ids]), torch.cat([y_docs[i] for i in train_doc_ids])
        x_eval, y_eval = torch.cat([x_docs[i] for i in eval_doc_ids]), torch.cat([y_docs[i] for i in eval_doc_ids])
        total_tokens = len(x_train) + len(x_eval)
        del x_docs, y_docs
        log("document_split_complete", layer=idx, train_documents=len(train_doc_ids),
            eval_documents=len(eval_doc_ids), train_tokens=len(x_train), eval_tokens=len(x_eval))
        moe = make_moe_from_ffn(ffn, a.num_experts, a.top_k).to(device)
        params = [q for q in moe.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=a.train_batch_size, shuffle=True)
        moe.train()
        for epoch in range(a.epochs):
            losses = []
            for step, (xb, yb) in enumerate(loader, 1):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred, balance_loss, _ = moe(xb, return_aux=True)
                target = yb.float()
                mse = F.mse_loss(pred.float(), target)
                relative_mse = mse / target.square().mean().detach().clamp_min(1e-8)
                loss = relative_mse + a.balance_loss_weight * balance_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                if step == 1 or step % 20 == 0 or step == len(loader):
                    log("layer_training_progress", layer=idx, epoch=epoch+1, total_epochs=a.epochs,
                        step=step, total_steps=len(loader), train_objective=losses[-1],
                        relative_mse=float(relative_mse.detach()), balance_loss=float(balance_loss.detach()))
            log("layer_epoch_complete", layer=idx, epoch=epoch+1,
                mean_train_mse=sum(losses)/max(1,len(losses)))
        moe.eval()
        eval_preds = []
        with torch.inference_mode():
            for start in range(0, len(x_eval), a.train_batch_size):
                pred = moe(x_eval[start:start+a.train_batch_size].to(device)).float().cpu()
                eval_preds.append(pred)
        eval_pred = torch.cat(eval_preds)
        eval_mse = F.mse_loss(eval_pred, y_eval).item()
        eval_relative_error = (eval_pred-y_eval).norm().item() / y_eval.norm().clamp_min(1e-12).item()
        eval_cosine = F.cosine_similarity(eval_pred.reshape(1,-1), y_eval.reshape(1,-1)).item()
        out = a.output_dir / f"layer_{idx:02d}_top{a.top_k}_{a.num_experts}experts.pt"
        torch.save({
            "model_id": a.model, "teacher_revision": teacher_revision, "layer": idx, "hidden_size": int(x_train.shape[-1]),
            "num_experts": a.num_experts, "top_k": a.top_k,
            "tokens_captured": int(total_tokens), "train_tokens": int(len(x_train)),
            "eval_tokens": int(len(x_eval)), "train_documents": len(train_doc_ids),
            "eval_documents": len(eval_doc_ids), "eval_mse": eval_mse,
            "eval_relative_error": eval_relative_error, "eval_cosine_similarity": eval_cosine,
            "balance_loss_weight": a.balance_loss_weight,
            "training_objective": "normalized FFN distillation loss plus Switch-style router load balancing",
            "state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()},
        }, out)
        log("layer_complete", layer=idx, eval_mse=eval_mse,
            eval_relative_error=eval_relative_error, eval_cosine_similarity=eval_cosine,
            checkpoint=str(out))
        del moe, opt, loader, x_train, y_train, x_eval, y_eval, params, eval_pred, eval_preds
        if device.type == "cuda":
            torch.cuda.empty_cache()
    present_checkpoints = sorted(
        path.name for path in a.output_dir.glob("layer_*_top*_experts.pt")
    )
    manifest = {
        "status": "layer_checkpoints_created",
        "model_id": a.model, "teacher_revision": teacher_revision, "device": str(device), "dtype": str(dtype),
        "num_layers_in_teacher": len(layers), "last_run_layer_range": [a.start_layer, last_layer],
        "num_experts": a.num_experts, "top_k": a.top_k, "epochs": a.epochs,
        "total_documents": len(texts), "train_documents": len(train_doc_ids),
        "eval_documents": len(eval_doc_ids), "eval_fraction": a.eval_fraction,
        "balance_loss_weight": a.balance_loss_weight, "max_length": a.max_length,
        "checkpoint_count_present": len(present_checkpoints),
        "checkpoints": present_checkpoints,
        "important_limitation": "Per-layer checkpoints only. This script does not yet package a loadable full MoE Transformers model or export Android runtime files.",
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log("all_layers_complete", manifest=str(a.output_dir / "manifest.json"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
