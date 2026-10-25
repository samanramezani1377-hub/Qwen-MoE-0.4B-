#!/usr/bin/env python3
"""Checkpoint-backed sparse FFN module and strict Qwen text-layer patching utilities.

This is an experimental runtime helper, not a Hugging Face AutoModel architecture.
The original Qwen model/config remain responsible for attention, embeddings and logits.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


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
    raise RuntimeError("Could not locate Qwen text decoder layers; inspect installed Transformers model structure.")


class CheckpointMoEFFN(nn.Module):
    """Module whose state-dict keys match the project's saved expert checkpoints."""
    def __init__(self, state_dict: dict[str, torch.Tensor], num_experts: int, top_k: int):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        gate_key = "experts.0.gate_proj.weight"
        down_key = "experts.0.down_proj.weight"
        router_key = "router.weight"
        for key in (gate_key, down_key, router_key):
            if key not in state_dict:
                raise ValueError(f"Checkpoint missing required tensor: {key}")
        hidden = int(state_dict[gate_key].shape[1])
        shard = int(state_dict[gate_key].shape[0])
        if tuple(state_dict[router_key].shape) != (num_experts, hidden):
            raise ValueError("Router shape does not match configured expert count/hidden size")
        self.hidden_size, self.num_experts, self.top_k = hidden, num_experts, top_k
        dtype = state_dict[gate_key].dtype
        device = state_dict[gate_key].device
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            prefix = f"experts.{i}."
            for suffix, expected in (
                ("gate_proj.weight", (shard, hidden)),
                ("up_proj.weight", (shard, hidden)),
                ("down_proj.weight", (hidden, shard)),
            ):
                key = prefix + suffix
                if key not in state_dict or tuple(state_dict[key].shape) != expected:
                    raise ValueError(f"Missing or invalid tensor {key}; expected shape {expected}")
            expert = nn.Module()
            expert.gate_proj = nn.Linear(hidden, shard, bias=False, device=device, dtype=dtype)
            expert.up_proj = nn.Linear(hidden, shard, bias=False, device=device, dtype=dtype)
            expert.down_proj = nn.Linear(shard, hidden, bias=False, device=device, dtype=dtype)
            self.experts.append(expert)
        self.router = nn.Linear(hidden, num_experts, bias=("router.bias" in state_dict),
                                device=device, dtype=torch.float32)
        missing, unexpected = self.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict checkpoint load failed; missing={missing}, unexpected={unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        scores = self.router(flat.float())
        selected_scores, selected_ids = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.softmax(selected_scores, dim=-1).to(flat.dtype)
        output = torch.zeros_like(flat)
        # This reference dispatch prioritizes correctness; it loops over all experts.
        # Optimize only after equivalence tests and target-device profiling.
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(selected_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_dtype = expert.gate_proj.weight.dtype
            expert_in = flat.index_select(0, token_ids).to(dtype=expert_dtype)
            contribution = expert.down_proj(F.silu(expert.gate_proj(expert_in)) * expert.up_proj(expert_in))
            contribution = contribution.to(flat.dtype) * weights[token_ids, slots].unsqueeze(-1)
            output = output.index_add(0, token_ids, contribution)
        return output.reshape(shape)


def patch_model_with_checkpoints(model: nn.Module, checkpoint_dir: str | Path,
                                 expected_layers: int = 24, num_experts: int = 32,
                                 top_k: int = 2, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Replace all decoder MLPs or fail closed; never silently leave a dense layer."""
    root = Path(checkpoint_dir)
    text_model = get_text_model(model)
    layers = text_model.layers
    if len(layers) != expected_layers:
        raise ValueError(f"Expected {expected_layers} text layers, found {len(layers)}")
    files = sorted(root.glob("layer_*_top*_experts.pt"))
    by_layer: dict[int, Path] = {}
    for path in files:
        import re
        match = re.fullmatch(r"layer_(\d+)_top(\d+)_(\d+)experts\.pt", path.name)
        if not match:
            continue
        layer_id, file_top_k, file_experts = map(int, match.groups())
        if file_top_k != top_k or file_experts != num_experts:
            continue
        if layer_id in by_layer:
            raise ValueError(f"Duplicate checkpoint for layer {layer_id}")
        by_layer[layer_id] = path
    missing = sorted(set(range(expected_layers)) - set(by_layer))
    if missing:
        raise FileNotFoundError(f"Missing compatible FFN checkpoints for layers: {missing}")
    reports = []
    for idx, layer in enumerate(layers):
        payload = torch.load(by_layer[idx], map_location=map_location, weights_only=False)
        for key, expected in (("layer", idx), ("num_experts", num_experts), ("top_k", top_k)):
            if payload.get(key) != expected:
                raise ValueError(f"{by_layer[idx]} metadata mismatch: {key}={payload.get(key)!r}, expected {expected}")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"{by_layer[idx]} has no state_dict")
        replacement = CheckpointMoEFFN(state, num_experts, top_k)
        original = getattr(layer, "mlp", None)
        if original is None:
            raise RuntimeError(f"Layer {idx} has no .mlp module")
        # Match the original layer's parameter device/dtype; router remains FP32.
        original_param = next(original.parameters())
        replacement = replacement.to(device=original_param.device)
        for expert in replacement.experts:
            expert.to(dtype=original_param.dtype)
        layer.mlp = replacement
        reports.append({"layer": idx, "checkpoint": str(by_layer[idx]),
                        "eval_relative_error": payload.get("eval_relative_error"),
                        "eval_cosine_similarity": payload.get("eval_cosine_similarity")})
    return {"layers_patched": len(reports), "num_experts": num_experts, "top_k": top_k,
            "checkpoints": reports, "status": "all_layers_patched"}
