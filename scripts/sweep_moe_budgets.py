#!/usr/bin/env python3
"""Compare active-parameter budgets and train tiny matched one-layer MoE variants.

This is a screening experiment, not a full-model conversion or reasoning benchmark.
The base Qwen model is loaded once; all expert counts use the same train/eval documents.
"""
from __future__ import annotations
import argparse, json, math, random, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from distill_real_qwen_ffn import MODEL_ID, get_text_model, make_moe_from_ffn

START = time.monotonic()

def log(event, **fields):
    print(json.dumps({"event": event, "elapsed_seconds": round(time.monotonic()-START, 2), **fields}, ensure_ascii=False), flush=True)

def capture_documents(model, ffn, tokenizer, docs, device, max_length, batch_docs):
    captured = []
    def hook(_module, inputs, output):
        captured.append((inputs[0].detach().float().cpu(), output.detach().float().cpu()))
    handle = ffn.register_forward_hook(hook)
    xs, ys = [], []
    with torch.inference_mode():
        total = (len(docs) + batch_docs - 1) // batch_docs
        for n, start in enumerate(range(0, len(docs), batch_docs), 1):
            batch = tokenizer(docs[start:start+batch_docs], return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length)
            model(**{k: v.to(device) for k, v in batch.items()})
            x, y = captured.pop()
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).bool()
            xs.append(x.reshape(-1, x.shape[-1])[mask.reshape(-1)])
            ys.append(y.reshape(-1, y.shape[-1])[mask.reshape(-1)])
            log("capture_progress", documents_processed=min(start+batch_docs, len(docs)),
                total_documents=len(docs), batch=n, total_batches=total)
    handle.remove()
    if captured:
        raise RuntimeError("Unexpected leftover activation batches")
    return torch.cat(xs), torch.cat(ys)

def quality(pred, target):
    p, y = pred.float(), target.float()
    mse = F.mse_loss(p, y).item()
    zero = F.mse_loss(torch.zeros_like(y), y).item()
    norm = torch.linalg.vector_norm(y).item()
    var = torch.var(y, unbiased=False).item()
    return {
        "mse": mse, "rmse": math.sqrt(max(mse, 0)),
        "zero_baseline_mse": zero,
        "relative_mse_improvement_vs_zero_pct": (1-mse/zero)*100 if zero else None,
        "normalized_mse_by_target_variance": mse/var if var else None,
        "relative_l2_error": torch.linalg.vector_norm(p-y).item()/norm if norm else None,
        "cosine_similarity": F.cosine_similarity(p.reshape(1,-1), y.reshape(1,-1), dim=1).item(),
        "target_rms": math.sqrt(max(torch.mean(y*y).item(), 0)),
        "prediction_rms": math.sqrt(max(torch.mean(p*p).item(), 0)),
    }

def timed(fn, batch, repeats):
    with torch.inference_mode():
        for _ in range(2): fn(batch)
        vals=[]
        for _ in range(repeats):
            t=time.perf_counter(); fn(batch); vals.append(time.perf_counter()-t)
    med=statistics.median(vals)
    return {"median_ms_per_batch": round(med*1000, 4),
            "token_vectors_per_second": round(len(batch)/med, 2) if med else None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--train-text-file", type=Path, required=True)
    p.add_argument("--eval-text-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--experts", default="8,16,32,48")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--batch-docs", type=int, default=2)
    p.add_argument("--max-train-token-vectors", type=int, default=1024)
    p.add_argument("--max-eval-token-vectors", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--benchmark-repeats", type=int, default=3)
    a=p.parse_args()
    experts=sorted(set(int(v.strip()) for v in a.experts.split(",") if v.strip()))
    if not experts or any(e < a.top_k for e in experts): p.error("Invalid expert counts/top-k")
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA requested but unavailable")
    device=torch.device(a.device)
    train_docs=[s.strip() for s in a.train_text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    eval_docs=[s.strip() for s in a.eval_text_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(train_docs)<10 or len(eval_docs)<5: raise SystemExit("Need >=10 train and >=5 held-out documents")
    if set(train_docs) & set(eval_docs): raise SystemExit("Train/eval document leakage detected")
    log("loading_model", model=a.model, device=str(device))
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    dtype=torch.bfloat16 if device.type=="cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model=AutoModel.from_pretrained(a.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    text_model=get_text_model(model)
    if not 0 <= a.layer < len(text_model.layers): raise ValueError("Invalid layer index")
    layers=text_model.layers
    mlps=[getattr(layer,"mlp",None) for layer in layers]
    if any(m is None for m in mlps): raise RuntimeError("Missing FFN module in one or more text layers")
    ffn=mlps[a.layer]
    log("capturing_train_activations", documents=len(train_docs))
    x_train,y_train=capture_documents(model,ffn,tokenizer,train_docs,device,a.max_length,a.batch_docs)
    log("capturing_eval_activations", documents=len(eval_docs))
    x_eval,y_eval=capture_documents(model,ffn,tokenizer,eval_docs,device,a.max_length,a.batch_docs)
    if len(x_train)<64 or len(x_eval)<32: raise RuntimeError("Not enough token vectors captured")
    cfg=getattr(text_model,"config",None)
    hidden=int(getattr(cfg,"hidden_size",x_train.shape[-1]))
    vocab=getattr(cfg,"vocab_size",None)
    embedding=getattr(text_model,"embed_tokens",None)
    embedding_count=sum(q.numel() for q in embedding.parameters()) if embedding is not None else 0
    num_layers=len(layers)
    intermediate_size=int(ffn.gate_proj.weight.shape[0])
    body_total=sum(q.numel() for q in text_model.parameters())
    ffn_counts=[sum(q.numel() for q in m.parameters()) for m in mlps]
    dense_ffn_total=sum(ffn_counts)
    non_ffn_core=body_total-embedding_count-dense_ffn_total
    head_count=int(vocab)*hidden if vocab else None
    tied=getattr(cfg,"tie_word_embeddings",None)
    del model, text_model, layers, mlps
    # Bound the experiment so the variants are comparable and CPU cost is predictable.
    g=torch.Generator().manual_seed(20261025)
    if len(x_train)>a.max_train_token_vectors:
        ids=torch.randperm(len(x_train), generator=g)[:a.max_train_token_vectors]
        x_train,y_train=x_train[ids],y_train[ids]
    if len(x_eval)>a.max_eval_token_vectors:
        ids=torch.randperm(len(x_eval), generator=torch.Generator().manual_seed(20261026))[:a.max_eval_token_vectors]
        x_eval,y_eval=x_eval[ids],y_eval[ids]
    ffn.eval()
    ffn_dtype=next(ffn.parameters()).dtype
    variants=[]
    for e in experts:
        torch.manual_seed(1700+e)
        try:
            moe=make_moe_from_ffn(ffn,e,a.top_k).to(device).eval()
        except ValueError as ex:
            log("variant_skipped", experts=e, reason=str(ex))
            variants.append({"num_experts":e,"status":"skipped","reason":str(ex)})
            continue
        opt=torch.optim.AdamW([q for q in moe.parameters() if q.requires_grad],lr=a.lr)
        moe.train()
        steps=0
        for epoch in range(a.epochs):
            perm=torch.randperm(len(x_train),generator=torch.Generator().manual_seed(9000+e*31+epoch))
            epoch_losses=[]
            for start in range(0,len(perm),a.train_batch_size):
                ids=perm[start:start+a.train_batch_size]
                xb=x_train[ids].to(device)
                yb=y_train[ids].to(device)
                opt.zero_grad(set_to_none=True)
                loss=F.mse_loss(moe(xb).float(),yb.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(moe.parameters(),1.0)
                opt.step()
                epoch_losses.append(float(loss.detach()))
                steps+=1
            log("variant_epoch_complete",experts=e,epoch=epoch+1,epochs=a.epochs,
                train_mse=sum(epoch_losses)/max(1,len(epoch_losses)),steps=steps)
        moe.eval()
        with torch.inference_mode():
            preds=[]
            for start in range(0,len(x_eval),128):
                preds.append(moe(x_eval[start:start+128].to(device)).float().cpu())
            pred=torch.cat(preds)
        q=quality(pred,y_eval)
        active_ffn=sum(c*a.top_k/e + (hidden*e+e) for c in ffn_counts)
        active_core=int(round(non_ffn_core+active_ffn))
        all_moe_total=int(round(body_total+num_layers*(hidden*e+e)))
        bench=[]
        for n in sorted(set([1,min(32,len(x_eval))])):
            batch=x_eval[:n].to(device=device,dtype=ffn_dtype)
            base=timed(lambda z:ffn(z),batch,a.benchmark_repeats)
            repl=timed(lambda z:moe(z),batch,a.benchmark_repeats)
            bench.append({"token_vectors":n,"dense_ffn":base,"trained_moe":repl,
                "throughput_ratio_moe_over_dense":round(repl["token_vectors_per_second"]/base["token_vectors_per_second"],4)
                    if base["token_vectors_per_second"] else None})
        record={"num_experts":e,"top_k":a.top_k,"expert_intermediate_size":intermediate_size//e if intermediate_size%e==0 else None,
            "status":"ok","train_token_vectors":len(x_train),"eval_token_vectors":len(x_eval),
            "epochs":a.epochs,"train_batch_size":a.train_batch_size,"training_steps":steps,
            "quality_vs_dense_teacher":q,"benchmark_single_ffn_only":bench,
            "parameter_budget":{"text_backbone_unique_parameters":int(body_total),
                "input_embedding_parameters":int(embedding_count),"dense_ffn_parameters_all_layers":int(dense_ffn_total),
                "non_ffn_core_estimate":int(non_ffn_core),"active_ffn_plus_routers_estimate":int(round(active_ffn)),
                "active_transformer_core_estimate":active_core,
                "active_core_plus_full_logits_projection_estimate":active_core+(head_count or 0),
                "all_layer_moe_unique_parameter_estimate":all_moe_total,
                "target_510M_core_met":active_core<=510_000_000,
                "target_450M_core_met":active_core<=450_000_000,
                "target_400M_core_met":active_core<=400_000_000,
                "target_510M_including_logits_met":active_core+(head_count or 0)<=510_000_000,
                "tie_word_embeddings":tied}}
        variants.append(record)
        log("variant_complete",experts=e,active_core_estimate=active_core,
            eval_mse=q["mse"],relative_l2_error=q["relative_l2_error"],
            cosine_similarity=q["cosine_similarity"],
            throughput_ratio=bench[0]["throughput_ratio_moe_over_dense"] if bench else None)
        del moe,opt
    report={"status":"ok","model":a.model,"layer":a.layer,"device":str(device),
        "train_documents":len(train_docs),"eval_documents":len(eval_docs),
        "train_document_hashes":[__import__("hashlib").sha256(s.encode()).hexdigest() for s in train_docs],
        "eval_document_hashes":[__import__("hashlib").sha256(s.encode()).hexdigest() for s in eval_docs],
        "document_overlap":0,"max_length":a.max_length,"top_k":a.top_k,
        "limits":{"max_train_token_vectors":a.max_train_token_vectors,"max_eval_token_vectors":a.max_eval_token_vectors},
        "parameter_floor_estimate_excluding_ffn":int(non_ffn_core),
        "vocab_size":vocab,"hidden_size":hidden,"full_logits_projection_parameters":head_count,
        "variants":variants,
        "interpretation":"One real FFN layer distilled on a bounded token sample; MSE measures layer-output imitation, not end-to-end language quality or reasoning. Active parameter counts are hypothetical all-layer estimates and exclude embedding lookup and report logits projection separately. Not Android performance."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","report":str(a.output),"variants":[{"experts":v.get("num_experts"),"status":v.get("status"),
        "active_core":v.get("parameter_budget",{}).get("active_transformer_core_estimate"),
        "mse":v.get("quality_vs_dense_teacher",{}).get("mse")} for v in variants]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
