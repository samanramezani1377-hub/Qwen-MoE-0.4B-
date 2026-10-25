#!/usr/bin/env python3
"""Validate dense-to-channel-sharded FFN equivalence with real Qwen layer-0 weights.

Only the three layer-0 text FFN tensors are fetched with HTTP Range requests.
The full checkpoint and all vision tensors are not downloaded.
"""

from __future__ import annotations

import json
import struct
import urllib.request

import torch
from torch.nn import functional as F

MODEL_ID = "Qwen/Qwen3.5-2B"
TARGET_KEYS = (
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
)
USER_AGENT = "Qwen-MoE-real-ffn-transfer-test/0.1"


def request_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range", "")
        if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(
                f"Server did not honor requested byte range {start}-{end}; "
                f"status={status}, Content-Range={content_range!r}"
            )
        data = response.read(end - start + 1)
    if len(data) != end - start + 1:
        raise RuntimeError(f"Truncated range: expected {end-start+1}, got {len(data)}")
    return data


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def read_header(url: str) -> tuple[int, dict]:
    prefix = request_range(url, 0, 7)
    header_size = struct.unpack("<Q", prefix)[0]
    if not 0 < header_size <= 32 * 1024 * 1024:
        raise RuntimeError(f"Unexpected safetensors header size: {header_size}")
    header_bytes = request_range(url, 8, 7 + header_size)
    return header_size, json.loads(header_bytes.decode("utf-8"))


def main() -> int:
    info = get_json(f"https://huggingface.co/api/models/{MODEL_ID}/revision/main?blobs=true")
    revision = info["sha"]
    base = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}"
    config = get_json(f"{base}/config.json")
    files = {item.get("rfilename"): item for item in info.get("siblings", [])}
    safetensors_name = next(
        (name for name in files if name.endswith(".safetensors")),
        None,
    )
    if not safetensors_name:
        raise RuntimeError("No safetensors checkpoint file found")
    checkpoint_size = files[safetensors_name].get("size")
    url = f"{base}/{safetensors_name}"
    header_size, header = read_header(url)

    missing = [key for key in TARGET_KEYS if key not in header]
    if missing:
        raise RuntimeError(f"Expected FFN tensors missing from header: {missing}")

    weights = {}
    downloaded_bytes = 16 + header_size
    data_start = 8 + header_size
    for key in TARGET_KEYS:
        meta = header[key]
        if meta.get("dtype") != "BF16":
            raise RuntimeError(f"Expected BF16 tensor for {key}, got {meta.get('dtype')}")
        begin, end = meta["data_offsets"]
        payload = request_range(url, data_start + begin, data_start + end - 1)
        downloaded_bytes += len(payload)
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(meta["shape"])
        weights[key] = tensor.to(dtype=torch.float32)

    gate = weights[TARGET_KEYS[0]]
    up = weights[TARGET_KEYS[1]]
    down = weights[TARGET_KEYS[2]]
    hidden_size = int(config["text_config"]["hidden_size"])
    intermediate_size = int(config["text_config"]["intermediate_size"])
    num_experts = 8
    expert_intermediate = intermediate_size // num_experts
    if intermediate_size % num_experts:
        raise RuntimeError("Intermediate size is not divisible by expert count")
    if tuple(gate.shape) != (intermediate_size, hidden_size):
        raise RuntimeError(f"Unexpected gate shape: {tuple(gate.shape)}")
    if tuple(up.shape) != (intermediate_size, hidden_size):
        raise RuntimeError(f"Unexpected up shape: {tuple(up.shape)}")
    if tuple(down.shape) != (hidden_size, intermediate_size):
        raise RuntimeError(f"Unexpected down shape: {tuple(down.shape)}")

    x = torch.randn(2, 3, hidden_size, dtype=torch.float32)
    dense = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    shard_outputs = []
    for expert_id in range(num_experts):
        start = expert_id * expert_intermediate
        stop = start + expert_intermediate
        hidden = F.silu(F.linear(x, gate[start:stop])) * F.linear(x, up[start:stop])
        shard_outputs.append(F.linear(hidden, down[:, start:stop]))
    reconstructed = torch.stack(shard_outputs, dim=0).sum(dim=0)
    max_abs_error = (dense - reconstructed).abs().max().item()
    if not torch.allclose(dense, reconstructed, atol=5e-4, rtol=5e-4):
        raise AssertionError(f"Real-weight dense-equivalence failed; max error={max_abs_error}")

    one_expert_params = 3 * hidden_size * expert_intermediate
    dense_params = 3 * hidden_size * intermediate_size
    all_expert_params = one_expert_params * num_experts
    active_top2_params = one_expert_params * 2
    result = {
        "status": "passed",
        "model_id": MODEL_ID,
        "resolved_revision": revision,
        "checkpoint_file": safetensors_name,
        "checkpoint_file_bytes": checkpoint_size,
        "header_bytes": header_size,
        "selected_tensor_keys": list(TARGET_KEYS),
        "selected_tensor_shapes": {key: list(header[key]["shape"]) for key in TARGET_KEYS},
        "selected_tensor_bytes_downloaded": sum(
            header[key]["data_offsets"][1] - header[key]["data_offsets"][0]
            for key in TARGET_KEYS
        ),
        "total_bytes_read_including_headers": downloaded_bytes,
        "download_scope": "only 3 FFN tensor payloads from text layer 0; no full checkpoint download",
        "dense_ffn_parameters_one_layer": dense_params,
        "one_expert_parameters": one_expert_params,
        "all_8_experts_parameters": all_expert_params,
        "active_top2_ffn_parameters_one_layer": active_top2_params,
        "dense_vs_8_shard_sum_max_abs_error": max_abs_error,
        "caveat": (
            "This validates only the mathematical channel-sharding mapping for layer 0. "
            "It does not validate top-2 quality, training, full-model loading, or Android runtime. "
            "Top-2 omits six shards and needs distillation/fine-tuning."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
