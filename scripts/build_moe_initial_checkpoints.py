#!/usr/bin/env python3
"""Create untrained, analysis-initialized per-layer MoE checkpoints for the single KD run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModel
from distill_real_qwen_ffn import get_text_model, make_moe_from_ffn

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="Qwen/Qwen3.5-2B")
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--num-experts",type=int,default=32)
    p.add_argument("--top-k",type=int,default=2)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8"))
    if plan.get("model_id")!=a.model: raise ValueError(f"Plan model {plan.get('model_id')} != requested {a.model}")
    if int(plan.get("num_experts",-1))!=a.num_experts: raise ValueError("Plan expert count mismatch")
    revision=plan.get("teacher_revision")
    if not revision: raise ValueError("Plan has no immutable teacher revision")
    model=AutoModel.from_pretrained(a.model,revision=revision,torch_dtype="auto",low_cpu_mem_usage=True).eval()
    if getattr(model.config,"_commit_hash",None)!=revision: raise ValueError("Loaded teacher revision differs from analysis plan")
    layers=get_text_model(model).layers
    if len(layers)!=int(plan["num_layers"]): raise ValueError("Model layer count differs from analysis plan")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for idx,layer in enumerate(layers):
        record=plan["layers"].get(str(idx))
        if not record: raise ValueError(f"Analysis plan missing layer {idx}")
        ffn=layer.mlp
        if int(record["intermediate_size"])!=int(ffn.gate_proj.weight.shape[0]): raise ValueError(f"Layer {idx} intermediate size mismatch")
        moe=make_moe_from_ffn(ffn,a.num_experts,a.top_k,channel_groups=record["channel_groups"])
        path=a.output_dir/f"layer_{idx:02d}_top{a.top_k}_{a.num_experts}experts.pt"
        payload={"model_id":a.model,"teacher_revision":revision,"layer":idx,
                 "hidden_size":int(record["hidden_size"]),"num_experts":a.num_experts,"top_k":a.top_k,
                 "expert_width":int(record["expert_width"]),"initialization":"pretrained FFN channel slices grouped by measured activation and projected weight signatures",
                 "channel_groups":record["channel_groups"],
                 "training_objective":"initialization only; next and only training route is ChatGPT text CE + pinned Qwen sparse-logit KD",
                 "state_dict":{k:v.detach().cpu() for k,v in moe.state_dict().items()}}
        torch.save(payload,path)
        print(json.dumps({"event":"initialized_checkpoint","layer":idx,"path":str(path),"revision":revision}),flush=True)
    manifest={"model_id":a.model,"teacher_revision":revision,"layers":len(layers),"num_experts":a.num_experts,"top_k":a.top_k,
              "status":"initialized_not_trained","next_step":"single ChatGPT text + Qwen sparse-logit KD training"}
    (a.output_dir/"initialization_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest),flush=True)
if __name__=="__main__": main()
