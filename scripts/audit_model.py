#!/usr/bin/env python3
"""Audit public Hugging Face model configuration without downloading model weights."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Qwen-MoE-0.4B-config-audit/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--output", default="audit-report.json", help="Output JSON path")
    args = parser.parse_args()

    base = f"https://huggingface.co/{args.model}/resolve/main"
    config = fetch_json(f"{base}/config.json")
    text = config.get("text_config", config)
    hidden = int(text.get("hidden_size", 0))
    vocab = int(text.get("vocab_size", config.get("vocab_size", 0)))
    intermediate = int(text.get("intermediate_size", 0))
    layers = int(text.get("num_hidden_layers", 0))
    experts = 8
    top_k = 2
    expert_intermediate = 768

    # SwiGLU gate/up/down linear weights, excluding biases and norms.
    per_expert = 3 * hidden * expert_intermediate
    active_ffn = per_expert * top_k * layers
    total_expert_ffn = per_expert * experts * layers
    tied_embedding = hidden * vocab if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)) else None
    dense_ffn = 3 * hidden * intermediate * layers

    report = {
        "model_id": args.model,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "transformers_version_hint": config.get("transformers_version"),
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "tie_word_embeddings": text.get("tie_word_embeddings", config.get("tie_word_embeddings")),
            "layer_types": text.get("layer_types"),
        },
        "estimates_excluding_attention_norms_biases_and_router": {
            "shared_embedding_parameters_if_tied": tied_embedding,
            "dense_ffn_parameters_all_layers_swiglu_estimate": dense_ffn,
            "moe_experts": experts,
            "top_k": top_k,
            "expert_intermediate_size": expert_intermediate,
            "parameters_per_expert_ffn": per_expert,
            "total_expert_ffn_parameters_all_layers": total_expert_ffn,
            "active_expert_ffn_parameters_per_token_all_layers": active_ffn,
            "warning": "Active expert FFN parameters are not total active model parameters; attention, embeddings/logits and routing also contribute.",
        },
        "scope": "Configuration-only audit. No model weights downloaded and no training performed.",
    }

    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Audit failed: {exc}")
