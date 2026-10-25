#!/usr/bin/env python3
"""CPU smoke test for splitting a dense SwiGLU FFN into channel-sharded experts.

This is a small synthetic unit test, not the full Qwen model. It proves that
summing all eight channel shards reproduces the dense FFN, and checks that the
top-2 sparse dispatch has valid shapes and gradients.
"""

from __future__ import annotations

import json
import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseSwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SwiGLUExpert(nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        self.up_proj = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False)
        self.down_proj = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(down_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ChannelShardedTopKMoE(nn.Module):
    def __init__(self, dense: DenseSwiGLUFFN, num_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        hidden = dense.gate_proj.in_features
        intermediate = dense.gate_proj.out_features
        if intermediate % num_experts:
            raise ValueError("Dense intermediate size must divide evenly across experts")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.num_experts = num_experts
        self.top_k = top_k
        shard = intermediate // num_experts
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    dense.gate_proj.weight[i * shard : (i + 1) * shard, :].detach().clone(),
                    dense.up_proj.weight[i * shard : (i + 1) * shard, :].detach().clone(),
                    dense.down_proj.weight[:, i * shard : (i + 1) * shard].detach().clone(),
                )
                for i in range(num_experts)
            ]
        )
        self.router = nn.Linear(hidden, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

    def sum_all_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Exact dense-equivalent path when every channel shard is summed."""
        return torch.stack([expert(x) for expert in self.experts], dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        scores = self.router(flat)
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        route_weights = torch.softmax(top_scores, dim=-1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(top_indices == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_ids))
            weighted = expert_output * route_weights[token_ids, slots].unsqueeze(-1)
            output = output.index_add(0, token_ids, weighted)
        return output.reshape(*original_shape[:-1], self.hidden_size)


def main() -> int:
    torch.manual_seed(17)
    hidden, intermediate, experts, top_k = 32, 64, 8, 2
    dense = DenseSwiGLUFFN(hidden, intermediate)
    moe = ChannelShardedTopKMoE(dense, num_experts=experts, top_k=top_k)
    x = torch.randn(5, 7, hidden)

    teacher = dense(x)
    all_expert_sum = moe.sum_all_experts(x)
    max_abs_error = (teacher - all_expert_sum).abs().max().item()
    if not torch.allclose(teacher, all_expert_sum, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"All-expert sum did not reproduce dense FFN: max error={max_abs_error}")

    sparse_output = moe(x)
    if sparse_output.shape != x.shape or not torch.isfinite(sparse_output).all():
        raise AssertionError("Top-k MoE output has an invalid shape or non-finite values")
    loss = sparse_output.square().mean()
    loss.backward()
    if moe.router.weight.grad is None or not torch.isfinite(moe.router.weight.grad).all():
        raise AssertionError("Router did not receive finite gradients")
    if any(p.grad is None for p in moe.parameters() if p.requires_grad):
        raise AssertionError("At least one MoE parameter did not receive a gradient")

    dense_params = sum(p.numel() for p in dense.parameters())
    expert_params = sum(p.numel() for e in moe.experts for p in e.parameters())
    active_ffn_per_token = 3 * hidden * (intermediate // experts) * top_k
    if dense_params != expert_params:
        raise AssertionError(f"Parameter budget mismatch: dense={dense_params}, experts={expert_params}")

    result = {
        "status": "passed",
        "test_scope": "synthetic one-layer CPU smoke test; not the Qwen checkpoint",
        "dense_intermediate": intermediate,
        "experts": experts,
        "expert_intermediate": intermediate // experts,
        "top_k": top_k,
        "dense_ffn_parameters": dense_params,
        "all_experts_ffn_parameters": expert_params,
        "active_expert_ffn_parameters_per_token_one_layer": active_ffn_per_token,
        "dense_vs_all_experts_max_abs_error": max_abs_error,
        "topk_output_shape": list(sparse_output.shape),
        "router_gradient_l1": moe.router.weight.grad.abs().sum().item(),
        "caveat": (
            "Channel sharding preserves the dense FFN only when outputs of all experts are summed. "
            "Top-2 routing drops other shards and therefore requires distillation/fine-tuning; this "
            "test does not prove quality or mobile performance."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
