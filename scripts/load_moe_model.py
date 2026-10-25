#!/usr/bin/env python3
"""Load base Qwen, patch all 24 text FFNs, and generate a test continuation.

This overlays experimental per-layer MoE checkpoints on base weights fetched from Hugging Face.
It is not a standalone Transformers AutoModel package.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_runtime import patch_model_with_checkpoints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    a = p.parse_args()
    if a.max_new_tokens < 1:
        p.error("max-new-tokens must be >=1")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    patch = patch_model_with_checkpoints(model, a.checkpoint_dir, expected_layers=24,
                                         num_experts=a.num_experts, top_k=a.top_k, map_location="cpu")
    inputs = tokenizer(a.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    print(json.dumps({"status":"generation_completed","patch_report":patch,
                      "prompt":a.prompt,"generated_text":tokenizer.decode(output[0],skip_special_tokens=True),
                      "warning":"Research runtime only; not a standalone AutoModel package or Android deployment."},
                     ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
