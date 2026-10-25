#!/usr/bin/env python3
"""Independent-document quality, baselines, parameter accounting and memory benchmark."""
from __future__ import annotations
import argparse, hashlib, json, math, resource, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from distill_real_qwen_ffn import MODEL_ID, get_text_model, make_moe_from_ffn

def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"): return round(int(line.split()[1])/1024,2)
    except OSError: pass
    return None
def peak_rss_mb():
    try: return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
    except Exception: return None
def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)
def metrics(pred,target):
    p,y=pred.float(),target.float()
    mse=F.mse_loss(p,y).item(); zero=F.mse_loss(torch.zeros_like(y),y).item()
    var=torch.var(y,unbiased=False).item(); ynorm=torch.linalg.vector_norm(y).item()
    return {"mse":mse,"rmse":math.sqrt(max(mse,0)),"zero_baseline_mse":zero,
      "mse_vs_zero_baseline_ratio":mse/zero if zero else None,
      "relative_mse_improvement_vs_zero_pct":(1-mse/zero)*100 if zero else None,
      "normalized_mse_by_target_variance":mse/var if var else None,
      "relative_l2_error":torch.linalg.vector_norm(p-y).item()/ynorm if ynorm else None,
      "cosine_similarity":F.cosine_similarity(p.reshape(1,-1),y.reshape(1,-1),dim=1).item(),
      "target_rms":math.sqrt(max(torch.mean(y*y).item(),0)),
      "target_std":math.sqrt(max(var,0)),"prediction_rms":math.sqrt(max(torch.mean(p*p).item(),0)),
      "max_absolute_error":torch.max(torch.abs(p-y)).item()}
def measure(fn,batch,device,warmup,repeats):
    with torch.inference_mode():
        for _ in range(warmup): fn(batch)
        sync(device); times=[]
        for _ in range(repeats):
            sync(device); t=time.perf_counter(); fn(batch); sync(device); times.append(time.perf_counter()-t)
    med=statistics.median(times)
    return {"batch_token_vectors":int(batch.shape[0]),"median_ms_per_batch":round(med*1000,4),
      "mean_ms_per_batch":round(statistics.mean(times)*1000,4),
      "median_ms_per_token_vector":round(med*1000/max(1,batch.shape[0]),4),
      "token_vectors_per_second":round(batch.shape[0]/med,2) if med else None,"warmup":warmup,"repeats":repeats}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=MODEL_ID); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--eval-text-file",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--layer",type=int,default=0); p.add_argument("--max-length",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--num-experts",type=int,default=32)
    p.add_argument("--top-k",type=int,default=2); p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=12); p.add_argument("--device",choices=("auto","cpu","cuda"),default="cpu")
    p.add_argument("--trust-remote-code",action="store_true"); a=p.parse_args()
    device=torch.device("cuda" if a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available()) else "cpu")
    if a.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    texts=[s.strip() for s in a.eval_text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(texts)<3: raise SystemExit(f"Need >=3 held-out documents, got {len(texts)}")
    if len({hashlib.sha256(t.encode()).hexdigest() for t in texts})!=len(texts):
        raise SystemExit("Duplicate evaluation documents; refusing misleading evaluation")
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    rss_start=rss_mb()
    dtype=torch.bfloat16 if device.type=="cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=a.trust_remote_code)
    model=AutoModel.from_pretrained(a.model,torch_dtype=dtype,low_cpu_mem_usage=True,
        trust_remote_code=a.trust_remote_code).to(device).eval()
    tm=get_text_model(model); layers=tm.layers
    if not 0<=a.layer<len(layers): raise ValueError(f"Layer {a.layer} invalid; model has {len(layers)} layers")
    mlps=[getattr(layer,"mlp",None) for layer in layers]
    ffn_counts=[sum(q.numel() for q in m.parameters()) if m is not None else 0 for m in mlps]
    body_total=sum(q.numel() for q in tm.parameters()); ffn_total=sum(ffn_counts)
    emb=getattr(tm,"embed_tokens",None); emb_count=sum(q.numel() for q in emb.parameters()) if emb is not None else 0
    cfg=getattr(tm,"config",None); vocab=getattr(cfg,"vocab_size",None); hidden=getattr(cfg,"hidden_size",None)
    tied=getattr(cfg,"tie_word_embeddings",None)
    head_count=int(vocab)*int(hidden) if vocab and hidden else None
    ffn=mlps[a.layer]
    if ffn is None: raise RuntimeError("Selected layer has no .mlp")
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    moe=make_moe_from_ffn(ffn,a.num_experts,a.top_k).to(device)
    missing,unexpected=moe.load_state_dict(ckpt["state_dict"],strict=True)
    if missing or unexpected: raise RuntimeError(f"Checkpoint mismatch missing={missing}, unexpected={unexpected}")
    random_moe=make_moe_from_ffn(ffn,a.num_experts,a.top_k).to(device).eval()
    random_moe.load_state_dict(ckpt["state_dict"],strict=True)
    torch.manual_seed(9281); nn.init.normal_(random_moe.router.weight,mean=0.0,std=0.01); nn.init.zeros_(random_moe.router.bias)
    captures=[]
    def hook(_m,inputs,out): captures.append((inputs[0].detach().float().cpu(),out.detach().float().cpu()))
    h=ffn.register_forward_hook(hook)
    with torch.inference_mode():
        for start in range(0,len(texts),a.batch_size):
            batch=tok(texts[start:start+a.batch_size],return_tensors="pt",padding=True,truncation=True,max_length=a.max_length)
            model(**{k:v.to(device) for k,v in batch.items()})
            mask=batch.get("attention_mask",torch.ones_like(batch["input_ids"])).bool().reshape(-1)
            xx,yy=captures[-1]; captures[-1]=(xx.reshape(-1,xx.shape[-1])[mask],yy.reshape(-1,yy.shape[-1])[mask])
    h.remove()
    x=torch.cat([z[0] for z in captures]); y=torch.cat([z[1] for z in captures])
    if len(x)<8: raise RuntimeError(f"Only {len(x)} valid token vectors captured")
    moe.eval(); random_moe.eval(); ffn.eval(); dt=next(ffn.parameters()).dtype
    with torch.inference_mode():
        pred=torch.cat([moe(x[i:i+128].to(device,dtype=dt)).float().cpu() for i in range(0,len(x),128)])
        rand=torch.cat([random_moe(x[i:i+128].to(device,dtype=dt)).float().cpu() for i in range(0,len(x),128)])
    quality={"trained_top2_moe":metrics(pred,y),"random_router_channel_sharded_moe":metrics(rand,y),
      "zero_output_baseline":{"mse":F.mse_loss(torch.zeros_like(y),y).item()}}
    router=int(hidden or 0)*a.num_experts+a.num_experts
    active_ffn=sum(c*a.top_k/a.num_experts+router for c in ffn_counts)
    active_core=int(round(body_total-emb_count-ffn_total+active_ffn))
    total_backbone_moe=int(round(body_total+len(layers)*router))
    params={"counted_from":"loaded Transformers text backbone and its actual layer modules",
      "text_backbone_total_unique_parameters":int(body_total),"dense_ffn_parameters_all_layers":int(ffn_total),
      "dense_ffn_parameters_per_layer":[int(v) for v in ffn_counts],"input_embedding_parameters":int(emb_count),
      "vocab_size":vocab,"hidden_size":hidden,"tie_word_embeddings":tied,"output_projection_parameters_for_logits":head_count,
      "full_model_unique_parameter_estimate":int(body_total+(0 if tied is True else (head_count or 0))),
      "num_layers":len(layers),"num_experts":a.num_experts,"top_k":a.top_k,
      "router_parameters_per_layer":router,"estimated_all_layer_moe_backbone_total_parameters":total_backbone_moe,
      "estimated_active_transformer_core_parameters_per_token_excluding_input_embedding_and_logits":active_core,
      "estimated_active_core_plus_full_logits_projection_parameters":active_core+(head_count or 0),
      "target_0_4B_active_transformer_core_met":active_core<=400_000_000,
      "target_0_4B_including_full_logits_projection_met":active_core+(head_count or 0)<=400_000_000,
      "definition":"Core active count excludes input embedding lookup matrix and output logits projection, adds top-k/num-experts of each FFN and one router per layer. Full-vocabulary logits projection is reported separately; tied weights count once in unique storage but are all used for logits.",
      "caveat":"Hypothetical all-layer conversion estimate only; checkpoint trained one FFN. Assumes each layer's FFN can be equally sharded."}
    bench=[]
    for n in sorted({1,min(32,len(x))}):
        b=x[:n].to(device,dtype=dt); base=measure(lambda z:ffn(z),b,device,a.warmup,a.repeats)
        repl=measure(lambda z:moe(z),b,device,a.warmup,a.repeats)
        bench.append({"batch_token_vectors":n,"original_ffn":base,"trained_top2_moe":repl,
          "moe_over_ffn_throughput_ratio":round(repl["token_vectors_per_second"]/base["token_vectors_per_second"],4) if base["token_vectors_per_second"] else None,
          "moe_over_ffn_latency_ratio":round(repl["median_ms_per_batch"]/base["median_ms_per_batch"],4) if base["median_ms_per_batch"] else None})
    mem={"rss_start_mb":rss_start,"rss_end_mb":rss_mb(),"process_peak_rss_mb":peak_rss_mb(),
      "note":"Process-wide peak includes model weights and activations; not isolated layer memory."}
    if device.type=="cuda":
        mem["cuda_peak_allocated_mb"]=round(torch.cuda.max_memory_allocated(device)/1024**2,2)
        mem["cuda_peak_reserved_mb"]=round(torch.cuda.max_memory_reserved(device)/1024**2,2)
    report={"status":"ok","model":a.model,"layer":a.layer,"device":str(device),
      "evaluation_documents":len(texts),"evaluation_token_vectors":int(len(x)),"independent_document_holdout":True,
      "quality_metrics":quality,"parameter_accounting":params,
      "benchmark":{"type":"single_ffn_layer_forward_only","warmup":a.warmup,"repeats":a.repeats,"results":bench,
        "includes_router_and_dispatch":True,"warning":"Not end-to-end generation or Android performance."},"memory":mem}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","evaluation_documents":len(texts),"evaluation_token_vectors":len(x),
      "trained_mse":quality["trained_top2_moe"]["mse"],"zero_baseline_mse":quality["zero_output_baseline"]["mse"],
      "relative_mse_improvement_vs_zero_pct":quality["trained_top2_moe"]["relative_mse_improvement_vs_zero_pct"],
      "active_core_estimate":active_core,"active_core_plus_logits_estimate":active_core+(head_count or 0),
      "peak_rss_mb":mem["process_peak_rss_mb"],"report":str(a.output)},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
