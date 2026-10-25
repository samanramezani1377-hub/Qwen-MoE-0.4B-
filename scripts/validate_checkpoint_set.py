#!/usr/bin/env python3
"""Validate that a directory contains compatible per-layer Qwen MoE checkpoints."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    pattern = re.compile(r"layer_(\d+)_top(\d+)_(\d+)experts\.pt")
    found = {}
    problems = []
    for path in sorted(a.checkpoint_dir.glob("layer_*.pt")):
        m = pattern.fullmatch(path.name)
        if not m:
            problems.append({"file": path.name, "error": "filename does not encode layer/top-k/expert count"})
            continue
        layer, top_k, experts = map(int, m.groups())
        if top_k != a.top_k or experts != a.num_experts:
            continue
        if layer in found:
            problems.append({"file": path.name, "error": f"duplicate layer {layer}"})
        found[layer] = path
    for layer in range(a.layers):
        if layer not in found:
            problems.append({"layer": layer, "error": "missing checkpoint"})
            continue
        path = found[layer]
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            expected = {"layer": layer, "top_k": a.top_k, "num_experts": a.num_experts}
            for key, value in expected.items():
                if obj.get(key) != value:
                    raise ValueError(f"{key}: got {obj.get(key)!r}, expected {value!r}")
            sd = obj.get("state_dict")
            if not isinstance(sd, dict):
                raise ValueError("state_dict missing")
            for key in ("experts.0.gate_proj.weight", "experts.0.up_proj.weight",
                        "experts.0.down_proj.weight", "router.weight"):
                if key not in sd:
                    raise ValueError(f"required tensor missing: {key}")
            hidden = sd["experts.0.gate_proj.weight"].shape[1]
            shard = sd["experts.0.gate_proj.weight"].shape[0]
            if tuple(sd["router.weight"].shape) != (a.num_experts, hidden):
                raise ValueError("router shape mismatch")
            for e in range(a.num_experts):
                prefix = f"experts.{e}."
                if tuple(sd[prefix+"gate_proj.weight"].shape) != (shard, hidden):
                    raise ValueError(f"expert {e} gate shape mismatch")
                if tuple(sd[prefix+"up_proj.weight"].shape) != (shard, hidden):
                    raise ValueError(f"expert {e} up shape mismatch")
                if tuple(sd[prefix+"down_proj.weight"].shape) != (hidden, shard):
                    raise ValueError(f"expert {e} down shape mismatch")
        except Exception as exc:
            problems.append({"file": path.name, "error": str(exc)})
    report = {"status": "passed" if not problems and len(found) == a.layers else "failed",
              "checkpoint_dir": str(a.checkpoint_dir), "expected_layers": a.layers,
              "compatible_checkpoint_count": len(found), "num_experts": a.num_experts,
              "top_k": a.top_k, "problems": problems,
              "note": "Structural validation only; does not prove language quality or full-model compatibility."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(rendered+"\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2

if __name__ == "__main__":
    raise SystemExit(main())
