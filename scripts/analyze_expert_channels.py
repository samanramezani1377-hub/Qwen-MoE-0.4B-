#!/usr/bin/env python3
"""Analyze real Qwen FFN weights and calibration activations, then create balanced expert channel groups.

This is an architecture-analysis/initialization step, not a training experiment. It does not
update the pretrained backbone. It creates an evidence-derived plan for the single downstream
ChatGPT-text + Qwen-logit-KD training route.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Any
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from distill_real_qwen_ffn import get_text_model

def balanced_bisect(features: torch.Tensor, indices: torch.Tensor, seed: int, rounds: int = 8):
    """Balanced 2-means split; recursive use produces equal-sized groups without dropping channels."""
    x = features.index_select(0, indices).float()
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randint(len(x), (1,), generator=g, device=x.device)].clone()
    d = torch.cdist(x, c).squeeze(1)
    c2 = x[d.argmax()].clone()
    for _ in range(rounds):
        axis = c2 - c
        projection = x @ axis.squeeze(0)
        order = torch.argsort(projection)
        left, right = order[:len(order)//2], order[len(order)//2:]
        c = x[left].mean(0, keepdim=True)
        c2 = x[right].mean(0, keepdim=True)
    order = torch.argsort(x @ (c2-c).squeeze(0))
    half = len(order)//2
    return indices[order[:half]], indices[order[half:]]

def recursive_groups(features: torch.Tensor, num_experts: int, seed: int):
    groups = [torch.arange(features.shape[0], device=features.device)]
    while len(groups) < num_experts:
        next_groups = []
        for i, group in enumerate(groups):
            left, right = balanced_bisect(features, group, seed + len(next_groups)*13 + i)
            next_groups.extend((left, right))
        groups = next_groups
    return [g.sort().values.cpu().tolist() for g in groups]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--text-file", type=Path, required=True, help="UTF-8 calibration corpus, one document per line")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-experts", type=int, default=32)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    p.add_argument("--seed", type=int, default=17)
    a=p.parse_args()
    if a.num_experts < 2 or (a.num_experts & (a.num_experts - 1)) != 0 or a.samples < 4 or a.max_length < 16:
        p.error("Require num-experts to be a power of two >=2, samples>=4 and max-length>=16")
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA requested but unavailable")
    device=torch.device("cuda" if a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available()) else "cpu")
    texts=[x.strip() for x in a.text_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(texts)<4: raise ValueError("Calibration text needs at least four non-empty documents")
    random.Random(a.seed).shuffle(texts); texts=texts[:a.samples]
    dtype=(torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16) if device.type=="cuda" else torch.bfloat16
    # Resolve a concrete Hub commit before loading, rather than relying on config._commit_hash
    # which can be absent with some multimodal AutoModel loaders.
    revision=HfApi().model_info(a.model).sha
    if not revision or len(revision) != 40:
        raise RuntimeError(f"Cannot resolve immutable Hugging Face commit for {a.model}; got {revision!r}")
    print(json.dumps({"event":"teacher_revision_resolved","model":a.model,"revision":revision}),flush=True)
    model=AutoModelForMultimodalLM.from_pretrained(a.model, revision=revision, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model, revision=revision)
    text_model=get_text_model(model); layers=text_model.layers
    # Calibration activation summaries from actual gate/up outputs; mask padding positions.
    sums={}; sums2={}; counts={}; handles=[]
    for li, layer in enumerate(layers):
        mlp=getattr(layer,"mlp",None)
        if mlp is None or not all(hasattr(mlp,k) for k in ("gate_proj","up_proj","down_proj")):
            raise RuntimeError(f"Layer {li} does not expose supported SwiGLU gate/up/down projections")
        for name in ("gate_proj","up_proj"):
            key=f"{li}:{name}"; sums[key]=None; sums2[key]=None; counts[key]=0
            def hook(_m,_inp,out,key=key):
                z=out.detach().float()
                if z.ndim==3: z=z.reshape(-1,z.shape[-1])
                # Batches are tokenized without padding below, so all rows are real tokens.
                part=z[:4096]
                s=part.sum(0).cpu(); s2=part.square().sum(0).cpu()
                sums[key]=s if sums[key] is None else sums[key]+s
                sums2[key]=s2 if sums2[key] is None else sums2[key]+s2
                counts[key]+=part.shape[0]
            handles.append(mlp._modules[name].register_forward_hook(hook))
    with torch.inference_mode():
        for i in range(0,len(texts),1):
            batch=tokenizer(texts[i:i+1],return_tensors="pt",truncation=True,max_length=a.max_length)
            model(**{k:v.to(device) for k,v in batch.items()})
            if (i+1)%25==0: print(json.dumps({"event":"activation_capture","documents":i+1,"total":len(texts)}),flush=True)
    for h in handles: h.remove()
    plan={"model_id":a.model,"teacher_revision":revision,"num_layers":len(layers),"num_experts":a.num_experts,
          "method":"balanced recursive bisection of projected gate/up/down weight signatures plus measured activation mean/std",
          "calibration_documents":len(texts),"max_length":a.max_length,"seed":a.seed,"layers":{}}
    for li,layer in enumerate(layers):
        mlp=layer.mlp; gate=mlp.gate_proj.weight.detach().float(); up=mlp.up_proj.weight.detach().float(); down=mlp.down_proj.weight.detach().float()
        intermediate,hidden=gate.shape
        if intermediate%a.num_experts: raise ValueError(f"Layer {li} intermediate={intermediate} is not divisible by {a.num_experts}")
        # Fixed seeded projections reduce feature size while retaining channel-specific weight signatures.
        gen=torch.Generator(device=gate.device).manual_seed(a.seed+li)
        proj=torch.randn(hidden,16,generator=gen,device=gate.device)/math.sqrt(16)
        weight_features=torch.cat([gate@proj,up@proj,down.T@proj],dim=1)
        stats=[]
        for name in ("gate_proj","up_proj"):
            key=f"{li}:{name}"; n=max(counts[key],1)
            mean=sums[key]/n; var=(sums2[key]/n-mean.square()).clamp_min(0)
            stats.extend([mean, var.sqrt()])
        activation_features=torch.stack(stats,dim=1).to(weight_features.device)
        features=torch.cat([torch.nn.functional.normalize(weight_features,dim=0),
                            torch.nn.functional.normalize(activation_features,dim=0)],dim=1)
        groups=recursive_groups(features,a.num_experts,a.seed+li*1009)
        flat=[v for group in groups for v in group]
        if len(flat)!=intermediate or len(set(flat))!=intermediate or min(flat)!=0 or max(flat)!=intermediate-1:
            raise RuntimeError(f"Layer {li}: groups do not partition all channels exactly")
        widths={len(g) for g in groups}
        if widths!={intermediate//a.num_experts}: raise RuntimeError(f"Layer {li}: expert groups are not balanced: {widths}")
        plan["layers"][str(li)]={"intermediate_size":intermediate,"hidden_size":hidden,
            "expert_width":intermediate//a.num_experts,"channel_groups":groups,
            "gate_activation_mean":float(sums[f"{li}:gate_proj"].mean().item()),
            "method_note":"groups are weight/activation-derived, balanced, and partition every original FFN channel exactly once"}
        print(json.dumps({"event":"layer_plan_complete","layer":li,"experts":a.num_experts,"expert_width":intermediate//a.num_experts}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"plan_saved","path":str(a.output),"teacher_revision":revision,"layers":len(layers)}),flush=True)
if __name__=="__main__": main()
