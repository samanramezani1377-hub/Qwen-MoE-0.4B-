#!/usr/bin/env python3
"""Audit a public Hugging Face model without downloading its weight tensors.

The report pins the config to the resolved Hub commit and records repository
file metadata plus parameter counts when the Hub API exposes them. It does not
claim exact tensor parameter counts if the Hub metadata does not provide them.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
USER_AGENT = "Qwen-MoE-config-audit/0.2"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def hub_url(path: str) -> str:
    return "https://huggingface.co/" + path.lstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit")
    parser.add_argument("--output", default="audit-report.json", help="Output JSON path")
    args = parser.parse_args()

    model_id = args.model.strip("/")
    info_url = hub_url(f"api/models/{model_id}/revision/{args.revision}?blobs=true")
    info = fetch_json(info_url)
    resolved_sha = info.get("sha")
    if not resolved_sha:
        raise RuntimeError("Hugging Face API did not return a resolved commit SHA.")

    base = hub_url(f"{model_id}/resolve/{resolved_sha}")
    config = fetch_json(f"{base}/config.json")
    tokenizer_config = None
    try:
        tokenizer_config = fetch_json(f"{base}/tokenizer_config.json")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        pass

    text_cfg = config.get("text_config", config)
    hidden = int(text_cfg.get("hidden_size", 0) or 0)
    vocab = int(text_cfg.get("vocab_size", config.get("vocab_size", 0)) or 0)
    intermediate = int(text_cfg.get("intermediate_size", 0) or 0)
    layers = int(text_cfg.get("num_hidden_layers", 0) or 0)
    tied = text_cfg.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    layer_types = text_cfg.get("layer_types", [])
    files = info.get("siblings", [])
    file_summary = [
        {
            "rfilename": item.get("rfilename"),
            "size": item.get("size"),
            "blob_id": item.get("blobId"),
        }
        for item in files
        if item.get("rfilename")
        and (
            item["rfilename"].endswith(".safetensors")
            or item["rfilename"] in {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "preprocessor_config.json",
            }
        )
    ]

    # This is a formula estimate for a SwiGLU FFN, not a count of the source checkpoint.
    experts, top_k, expert_intermediate = 8, 2, 768
    params_per_expert = 3 * hidden * expert_intermediate
    total_expert_ffn = params_per_expert * experts * layers
    active_expert_ffn = params_per_expert * top_k * layers
    dense_ffn = 3 * hidden * intermediate * layers
    hub_safetensors = info.get("safetensors")
    parameter_metadata = (
        hub_safetensors.get("parameters")
        if isinstance(hub_safetensors, dict)
        else None
    )

    report = {
        "model_id": model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_sha,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "model_api": info_url,
            "pinned_config": f"{base}/config.json",
            "pinned_tokenizer_config": f"{base}/tokenizer_config.json",
        },
        "license": info.get("license") or info.get("cardData", {}).get("license"),
        "pipeline_tag": info.get("pipeline_tag"),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "transformers_version_hint": config.get("transformers_version"),
        "text_config": {
            "model_type": text_cfg.get("model_type"),
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "tie_word_embeddings": tied,
            "layer_types": layer_types,
            "layer_type_counts": {
                name: layer_types.count(name) for name in sorted(set(layer_types))
            },
        },
        "tokenizer_config": {
            "tokenizer_class": (tokenizer_config or {}).get("tokenizer_class"),
            "model_max_length": (tokenizer_config or {}).get("model_max_length"),
            "chat_template_present": bool((tokenizer_config or {}).get("chat_template")),
        },
        "hub_parameter_metadata": parameter_metadata,
        "hub_safetensors_metadata": hub_safetensors,
        "relevant_files": file_summary,
        "formula_estimates_not_checkpoint_counts": {
            "embedding_matrix_parameters": hidden * vocab if hidden and vocab else None,
            "embedding_and_lm_head_tied": tied,
            "dense_swiglu_ffn_all_layers": dense_ffn,
            "moe_experts_per_converted_layer": experts,
            "top_k": top_k,
            "expert_intermediate_size": expert_intermediate,
            "parameters_per_expert_ffn": params_per_expert,
            "all_expert_ffn_parameters_all_layers": total_expert_ffn,
            "active_expert_ffn_parameters_per_token_all_layers": active_expert_ffn,
            "warning": (
                "Formula estimates only. They exclude attention, norms, biases, "
                "router and other components; active FFN is not total active parameters."
            ),
        },
        "scope": (
            "Metadata/config audit only. No weight tensor files were downloaded, "
            "no model was instantiated, and no training or paid compute was run. "
            "Hub parameter metadata is reported only if supplied by the API; exact "
            "checkpoint tensor counts still require inspecting safetensors headers "
            "or loading model weights."
        ),
    }

    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(f"Audit failed: {exc}")
