#!/usr/bin/env python3
"""Real-activation, one-layer Qwen FFN -> top-2 channel-sharded MoE distillation.

Example: python scripts/distill_real_qwen_ffn.py --text-file data/fa_train.txt --layer 0 --samples 256 --epochs 3 --output artifacts/layer0_top2.pt
Requires a UTF-8 text corpus with one sample per line. Loading Qwen3.5-2B needs several GB RAM.
No paid GPU is provisioned. This produces a research checkpoint for ONE FFN, not a complete model.
"""
from __future__ import annotations
import argparse, json, random, statistics, time
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-2B"
START_TIME = time.monotonic()


def progress(event: str, **fields: Any) -> None:
    payload = {"event": event, "elapsed_seconds": round(time.monotonic() - START_TIME, 2), **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_forward(fn, batch: torch.Tensor, device: torch.device,
                     warmup: int, repeats: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(batch)
        _sync(device)
        samples = []
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            fn(batch)
            _sync(device)
            samples.append(time.perf_counter() - started)
    median_seconds = statistics.median(samples)
    mean_seconds = statistics.mean(samples)
    count = int(batch.shape[0])
    return {
        "batch_tokens": count,
        "median_ms_per_batch": round(median_seconds * 1000, 4),
        "mean_ms_per_batch": round(mean_seconds * 1000, 4),
        "median_ms_per_token": round(median_seconds * 1000 / count, 4),
        "tokens_per_second": round(count / median_seconds, 2) if median_seconds else 0.0,
        "repeats": repeats,
        "warmup": warmup,
    }


def benchmark_ffn_vs_moe(ffn: nn.Module, moe: nn.Module, samples: torch.Tensor,
                         device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    """Compare one FFN and its MoE replacement on identical token vectors."""
    ffn.eval()
    moe.eval()
    n = int(samples.shape[0])
    sizes = sorted({1, min(32, n)})
    ffn_dtype = next(ffn.parameters()).dtype
    results = []
    route_counts = None
    for batch_size in sizes:
        batch = samples[:batch_size].to(device=device, dtype=ffn_dtype)
        progress("benchmark_started", batch_tokens=batch_size, warmup=warmup, repeats=repeats)
        base = _measure_forward(lambda value: ffn(value), batch, device, warmup, repeats)
        replacement = _measure_forward(lambda value: moe(value), batch, device, warmup, repeats)
        base_tps, moe_tps = base["tokens_per_second"], replacement["tokens_per_second"]
        results.append({
            "batch_tokens": batch_size,
            "same_input_and_dtype": True,
            "original_ffn": base,
            "top2_moe": replacement,
            "moe_over_ffn_throughput_ratio": round(moe_tps / base_tps, 4) if base_tps else None,
            "moe_over_ffn_latency_ratio": round(
                replacement["median_ms_per_batch"] / base["median_ms_per_batch"], 4
            ) if base["median_ms_per_batch"] else None,
        })
        progress("benchmark_batch_complete", batch_tokens=batch_size,
                 original_tokens_per_second=base_tps, moe_tokens_per_second=moe_tps,
                 moe_over_ffn_throughput_ratio=results[-1]["moe_over_ffn_throughput_ratio"])
    route_batch = samples[:min(128, n)].to(device=device, dtype=ffn_dtype)
    with torch.inference_mode():
        _, route_indices = torch.topk(moe.router(route_batch.float()), moe.top_k, dim=-1)
        route_counts = torch.bincount(route_indices.reshape(-1), minlength=moe.num_experts).cpu().tolist()
    return {
        "benchmark_type": "single_ffn_layer_forward_only",
        "device": str(device),
        "input_tokens_available": n,
        "comparison": results,
        "expert_route_assignments": route_counts,
        "scope_note": (
            "Measures FFN token-vector throughput only, including MoE routing/dispatch overhead. "
            "It is not end-to-end language-model generation speed or Android-device performance."
        ),
    }

def get_text_model(model: nn.Module) -> nn.Module:
    for path in (("language_model",), ("model", "language_model"), ("model",)):
        current: Any = model
        try:
            for part in path:
                current = getattr(current, part)
            if hasattr(current, "layers"):
                return current
        except AttributeError:
            pass
    raise RuntimeError("Could not locate text decoder layers in this Transformers/model version.")

def make_moe_from_ffn(ffn: nn.Module, num_experts: int, top_k: int) -> nn.Module:
    if not all(hasattr(ffn, n) for n in ("gate_proj", "up_proj", "down_proj")):
        raise TypeError(f"Expected SwiGLU gate/up/down FFN; got {type(ffn).__name__}")
    gate, up, down = (ffn.gate_proj.weight.detach(), ffn.up_proj.weight.detach(), ffn.down_proj.weight.detach())
    intermediate, hidden = gate.shape
    if intermediate % num_experts:
        raise ValueError(f"intermediate size {intermediate} not divisible by num_experts={num_experts}")
    shard, device, dtype = intermediate // num_experts, gate.device, gate.dtype

    class Expert(nn.Module):
        def __init__(self, i: int):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, shard, bias=False, device=device, dtype=dtype)
            self.up_proj = nn.Linear(hidden, shard, bias=False, device=device, dtype=dtype)
            self.down_proj = nn.Linear(shard, hidden, bias=False, device=device, dtype=dtype)
            lo, hi = i * shard, (i + 1) * shard
            with torch.no_grad():
                self.gate_proj.weight.copy_(gate[lo:hi])
                self.up_proj.weight.copy_(up[lo:hi])
                self.down_proj.weight.copy_(down[:, lo:hi])
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class TopKMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size, self.num_experts, self.top_k = hidden, num_experts, top_k
            self.experts = nn.ModuleList([Expert(i) for i in range(num_experts)])
            self.router = nn.Linear(hidden, num_experts, bias=True, device=device, dtype=torch.float32)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.router.bias)
        def forward(self, x):
            shape = x.shape
            flat = x.reshape(-1, self.hidden_size)
            scores = self.router(flat.float())
            top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)
            weights = torch.softmax(top_scores, dim=-1).to(flat.dtype)
            output = torch.zeros_like(flat)
            for i, expert in enumerate(self.experts):
                token_ids, slots = torch.where(top_indices == i)
                if token_ids.numel():
                    expert_dtype = next(expert.parameters()).dtype
                    expert_in = flat.index_select(0, token_ids).to(dtype=expert_dtype)
                    expert_out = expert(expert_in).to(dtype=flat.dtype)
                    output = output.index_add(0, token_ids, expert_out * weights[token_ids, slots].unsqueeze(-1))
            return output.reshape(shape)
    return TopKMoE()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-file", type=Path, required=True, help="UTF-8 text, one sample per line")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=Path("artifacts/layer0_top2.pt"))
    p.add_argument("--benchmark-output", type=Path, default=None,
                   help="JSON benchmark path (default: output filename with .benchmark.json suffix)")
    p.add_argument("--benchmark-warmup", type=int, default=3)
    p.add_argument("--benchmark-repeats", type=int, default=12)
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    if a.samples < 4 or a.epochs < 1 or a.batch_size < 1 or not 1 <= a.top_k <= a.num_experts:
        p.error("Require samples>=4, epochs/batch-size>=1 and 1<=top-k<=num-experts")
    if a.benchmark_warmup < 0 or a.benchmark_repeats < 1:
        p.error("Require benchmark-warmup>=0 and benchmark-repeats>=1")
    if a.benchmark_output is None:
        a.benchmark_output = a.output.with_suffix(".benchmark.json")
    progress("run_started", model=a.model, layer=a.layer, samples=a.samples, epochs=a.epochs,
             batch_size=a.batch_size, device_requested=a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available; no GPU will be provisioned.")
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(a.seed); torch.manual_seed(a.seed)
    texts = [s.strip() for s in a.text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(texts) < 4:
        raise ValueError("Text file must contain at least 4 non-empty lines.")
    random.Random(a.seed).shuffle(texts)
    texts = texts[:a.samples]
    progress("corpus_loaded", available_nonempty_lines=len(texts), text_file=str(a.text_file))
    progress("tokenizer_loading", model=a.model)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    # Keep the large base checkpoint near its native low-precision size on CPU too.
    # FFN activations are captured as FP32 for stable regression; expert weights stay low precision.
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.bfloat16
    progress("base_model_loading", model=a.model, dtype=str(dtype), device=str(device))
    model = AutoModel.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True,
                                      trust_remote_code=a.trust_remote_code).to(device).eval()
    progress("base_model_loaded", device=str(device))
    text_model = get_text_model(model)
    if not 0 <= a.layer < len(text_model.layers):
        raise ValueError(f"Layer index out of range; model has {len(text_model.layers)} layers.")
    ffn = getattr(text_model.layers[a.layer], "mlp", None)
    if ffn is None or not all(hasattr(ffn, n) for n in ("gate_proj", "up_proj", "down_proj")):
        raise RuntimeError("Selected layer does not expose a SwiGLU .mlp with gate_proj/up_proj/down_proj.")

    captures, masks = [], []
    def hook(_module, inputs, output):
        if inputs and inputs[0].ndim == 3 and output.ndim == 3:
            captures.append((inputs[0].detach().float().cpu(), output.detach().float().cpu()))
    handle = ffn.register_forward_hook(hook)
    capture_started = time.monotonic()
    capture_batches = (len(texts) + a.batch_size - 1) // a.batch_size
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, len(texts), a.batch_size), start=1):
            batch = tok(texts[start:start+a.batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=a.max_length)
            masks.append(batch.get("attention_mask", torch.ones_like(batch["input_ids"])).bool())
            model(**{k: v.to(device) for k, v in batch.items()})
            elapsed = max(time.monotonic() - capture_started, 1e-6)
            rate = batch_no / elapsed
            progress("activation_capture_progress", batch=batch_no, total_batches=capture_batches,
                     samples_processed=min(start + a.batch_size, len(texts)), total_samples=len(texts),
                     batches_per_second=round(rate, 3),
                     eta_seconds=round((capture_batches - batch_no) / rate, 1))
    handle.remove()
    progress("activation_capture_complete", batches=len(captures), samples=len(texts))
    if len(captures) != len(masks) or not captures:
        raise RuntimeError("FFN activation hook did not capture exactly one output per input batch.")
    x_parts, y_parts, mask_parts = [], [], []
    for (x, y), mask in zip(captures, masks):
        x_parts.append(x.reshape(-1, x.shape[-1])); y_parts.append(y.reshape(-1, y.shape[-1]))
        mask_parts.append(mask.reshape(-1))
    x_all, y_all, valid = torch.cat(x_parts), torch.cat(y_parts), torch.cat(mask_parts)
    x_all, y_all = x_all[valid], y_all[valid]
    progress("activations_ready", tokens_captured=int(len(x_all)), hidden_size=int(x_all.shape[-1]))
    del model, text_model
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(a.seed))
    n_eval = max(1, int(0.1 * len(perm)))
    eval_ids, train_ids = perm[:n_eval], perm[n_eval:]
    x_train, y_train = x_all[train_ids], y_all[train_ids]
    x_eval, y_eval = x_all[eval_ids], y_all[eval_ids]
    moe = make_moe_from_ffn(ffn, a.num_experts, a.top_k).to(device)
    progress("moe_initialized", experts=a.num_experts, top_k=a.top_k,
             train_tokens=int(len(train_ids)), eval_tokens=int(len(eval_ids)))
    params = [q for q in moe.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=a.batch_size, shuffle=True)
    total_steps = a.epochs * len(loader)
    train_step = 0
    training_started = time.monotonic()
    moe.train()
    for epoch in range(a.epochs):
        losses = []
        progress("epoch_started", epoch=epoch + 1, total_epochs=a.epochs, batches_per_epoch=len(loader))
        for batch_no, (xb, yb) in enumerate(loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(moe(xb).float(), yb.float())
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(loss.detach().item())
            train_step += 1
            elapsed = max(time.monotonic() - training_started, 1e-6)
            steps_per_second = train_step / elapsed
            progress("training_progress", epoch=epoch + 1, total_epochs=a.epochs,
                     batch=batch_no, total_batches=len(loader), step=train_step, total_steps=total_steps,
                     train_mse=round(losses[-1], 8), steps_per_second=round(steps_per_second, 3),
                     eta_seconds=round((total_steps - train_step) / steps_per_second, 1))
        epoch_mse = sum(losses) / max(1, len(losses))
        progress("epoch_complete", epoch=epoch + 1, train_mse=epoch_mse)
    moe.eval(); eval_losses = []
    eval_batches = (len(x_eval) + a.batch_size - 1) // a.batch_size
    with torch.inference_mode():
        for batch_no, i in enumerate(range(0, len(x_eval), a.batch_size), start=1):
            pred = moe(x_eval[i:i+a.batch_size].to(device)).float().cpu()
            eval_losses.append(F.mse_loss(pred, y_eval[i:i+a.batch_size]).item())
            progress("evaluation_progress", batch=batch_no, total_batches=eval_batches,
                     eval_tokens_processed=min(i + a.batch_size, len(x_eval)), total_eval_tokens=len(x_eval))
    eval_mse = sum(eval_losses) / max(1, len(eval_losses))
    progress("benchmark_stage_started", purpose="original FFN versus trained top-2 MoE")
    benchmark = benchmark_ffn_vs_moe(ffn, moe, x_all, device,
                                     warmup=a.benchmark_warmup, repeats=a.benchmark_repeats)
    benchmark.update({
        "model_id": a.model, "layer": a.layer, "num_experts": a.num_experts, "top_k": a.top_k,
        "eval_mse": eval_mse, "tokens_captured": int(len(x_all)),
        "training_corpus": str(a.text_file), "benchmark_repeats": a.benchmark_repeats,
    })
    a.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    a.benchmark_output.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    progress("benchmark_saved", path=str(a.benchmark_output))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": a.model, "layer": a.layer, "hidden_size": int(x_all.shape[-1]),
                "num_experts": a.num_experts, "top_k": a.top_k, "tokens_captured": len(x_all),
                "train_tokens": len(train_ids), "eval_tokens": len(eval_ids), "eval_mse": eval_mse,
                "benchmark_output": str(a.benchmark_output),
                "expert_route_assignments": benchmark.get("expert_route_assignments"),
                "state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()}}, a.output)
    progress("checkpoint_saved", path=str(a.output), eval_mse=eval_mse)
    print(json.dumps({"status": "ok", "model": a.model, "layer": a.layer, "device": str(device),
                      "tokens_captured": len(x_all), "train_tokens": len(train_ids), "eval_tokens": len(eval_ids),
                      "eval_mse": eval_mse, "checkpoint": str(a.output),
                      "benchmark": str(a.benchmark_output),
                      "note": "Single FFN layer prototype only; throughput is not full-model text generation speed."},
                     ensure_ascii=False, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
