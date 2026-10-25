#!/usr/bin/env python3
"""Real-activation, one-layer Qwen FFN -> top-2 channel-sharded MoE distillation.

Example: python scripts/distill_real_qwen_ffn.py --text-file data/fa_train.txt --layer 0 --samples 256 --epochs 3 --output artifacts/layer0_top2.pt
Requires a UTF-8 text corpus with one sample per line. Loading Qwen3.5-2B needs several GB RAM.
No paid GPU is provisioned. This produces a research checkpoint for ONE FFN, not a complete model.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-2B"

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
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    if a.samples < 4 or a.epochs < 1 or a.batch_size < 1 or not 1 <= a.top_k <= a.num_experts:
        p.error("Require samples>=4, epochs/batch-size>=1 and 1<=top-k<=num-experts")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available; no GPU will be provisioned.")
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    random.seed(a.seed); torch.manual_seed(a.seed)
    texts = [s.strip() for s in a.text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(texts) < 4:
        raise ValueError("Text file must contain at least 4 non-empty lines.")
    random.Random(a.seed).shuffle(texts)
    texts = texts[:a.samples]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    # Keep the large base checkpoint near its native low-precision size on CPU too.
    # FFN activations are captured as FP32 for stable regression; expert weights stay low precision.
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.bfloat16
    model = AutoModel.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True,
                                      trust_remote_code=a.trust_remote_code).to(device).eval()
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
    with torch.inference_mode():
        for start in range(0, len(texts), a.batch_size):
            batch = tok(texts[start:start+a.batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=a.max_length)
            masks.append(batch.get("attention_mask", torch.ones_like(batch["input_ids"])).bool())
            model(**{k: v.to(device) for k, v in batch.items()})
    handle.remove()
    if len(captures) != len(masks) or not captures:
        raise RuntimeError("FFN activation hook did not capture exactly one output per input batch.")
    x_parts, y_parts, mask_parts = [], [], []
    for (x, y), mask in zip(captures, masks):
        x_parts.append(x.reshape(-1, x.shape[-1])); y_parts.append(y.reshape(-1, y.shape[-1]))
        mask_parts.append(mask.reshape(-1))
    x_all, y_all, valid = torch.cat(x_parts), torch.cat(y_parts), torch.cat(mask_parts)
    x_all, y_all = x_all[valid], y_all[valid]
    del model, text_model
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(a.seed))
    n_eval = max(1, int(0.1 * len(perm)))
    eval_ids, train_ids = perm[:n_eval], perm[n_eval:]
    x_train, y_train = x_all[train_ids], y_all[train_ids]
    x_eval, y_eval = x_all[eval_ids], y_all[eval_ids]
    moe = make_moe_from_ffn(ffn, a.num_experts, a.top_k).to(device)
    params = [q for q in moe.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=a.batch_size, shuffle=True)
    moe.train()
    for epoch in range(a.epochs):
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(moe(xb).float(), yb.float())
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(loss.detach().item())
        print(json.dumps({"epoch": epoch + 1, "train_mse": sum(losses)/max(1, len(losses))}))
    moe.eval(); eval_losses = []
    with torch.no_grad():
        for i in range(0, len(x_eval), a.batch_size):
            pred = moe(x_eval[i:i+a.batch_size].to(device)).float().cpu()
            eval_losses.append(F.mse_loss(pred, y_eval[i:i+a.batch_size]).item())
    eval_mse = sum(eval_losses) / max(1, len(eval_losses))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": a.model, "layer": a.layer, "hidden_size": int(x_all.shape[-1]),
                "num_experts": a.num_experts, "top_k": a.top_k, "tokens_captured": len(x_all),
                "train_tokens": len(train_ids), "eval_tokens": len(eval_ids), "eval_mse": eval_mse,
                "state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()}}, a.output)
    print(json.dumps({"status": "ok", "model": a.model, "layer": a.layer, "device": str(device),
                      "tokens_captured": len(x_all), "train_tokens": len(train_ids), "eval_tokens": len(eval_ids),
                      "eval_mse": eval_mse, "checkpoint": str(a.output),
                      "note": "Single FFN layer prototype only; not a full converted model or a quality benchmark."},
                     ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
