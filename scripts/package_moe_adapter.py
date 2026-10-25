#!/usr/bin/env python3
"""Create a reproducible overlay package from validated per-layer MoE checkpoints.

The package contains checkpoint files, base-model reference, hashes, and loader scripts.
It deliberately does not duplicate base model weights; this is not a standalone model.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--model-revision", default="main", help="Prefer an immutable resolved commit SHA for releases")
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--layers", type=int, default=24)
    a = p.parse_args()
    if a.output_dir.resolve() == a.checkpoint_dir.resolve():
        p.error("output-dir must differ from checkpoint-dir")
    import subprocess, sys
    validator = Path(__file__).with_name("validate_checkpoint_set.py")
    check = subprocess.run([sys.executable,str(validator),"--checkpoint-dir",str(a.checkpoint_dir),
                            "--layers",str(a.layers),"--num-experts",str(a.num_experts),"--top-k",str(a.top_k)],
                           text=True,capture_output=True)
    if check.returncode:
        print(check.stdout)
        raise SystemExit("Checkpoint validation failed; package was not created.")
    files = sorted(a.checkpoint_dir.glob(f"layer_*_top{a.top_k}_{a.num_experts}experts.pt"))
    out = a.output_dir
    out.mkdir(parents=True,exist_ok=True)
    target = out / "checkpoints"
    target.mkdir(exist_ok=True)
    manifest = {"format":"qwen-moe-overlay-v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
                "base_model":a.model,"base_model_revision":a.model_revision,
                "layers":a.layers,"num_experts":a.num_experts,"top_k":a.top_k,
                "checkpoint_files":[],"validation":"structural validation passed",
                "standalone":False,
                "limitations":["Base model weights are not included.","Requires compatible Transformers support for Qwen3.5.",
                               "Not a Hugging Face AutoModel package and not an Android runtime export."]}
    for src in files:
        dst=target/src.name
        shutil.copy2(src,dst)
        manifest["checkpoint_files"].append({"name":dst.name,"size_bytes":dst.stat().st_size,"sha256":sha256(dst)})
    for name in ("moe_runtime.py","load_moe_model.py","evaluate_full_model.py","validate_checkpoint_set.py"):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    (out/"overlay_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# Experimental Qwen MoE overlay\n\n"
        "Contains per-layer sparse FFN checkpoints and runtime/evaluation helpers. "
        "Does not contain base Qwen weights and is not a standalone Transformers model.\n\n"
        f"- Base model: {a.model}\n- Revision: {a.model_revision}\n"
        f"- Layers: {a.layers}\n- Experts/layer: {a.num_experts}\n- Top-k: {a.top_k}\n\n"
        "Validate before use. See overlay_manifest.json for checkpoint SHA-256 hashes and limitations. "
        "Use load_moe_model.py after installing compatible torch/transformers.\n",
        encoding="utf-8")
    print(json.dumps({"status":"package_created","output_dir":str(out),"checkpoints":len(files),
                      "manifest":str(out/"overlay_manifest.json"),"standalone":False},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
