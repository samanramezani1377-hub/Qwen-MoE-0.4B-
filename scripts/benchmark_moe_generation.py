#!/usr/bin/env python3
"""End-to-end generation latency benchmark for a patched Qwen MoE overlay.

Run this on the actual target device for device-specific claims. Colab measurements are
not Android measurements. Records prefill latency, first-token latency, decode throughput,
and peak CUDA memory when available.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_runtime import patch_model_with_checkpoints


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Qwen/Qwen3.5-2B")
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--prompt",default="توضیح بده چرا برای حفظ استدلال باید یک مدل زبانی را با دادهٔ ارزیابی مستقل آزمایش کرد.")
    p.add_argument("--new-tokens",type=int,default=64)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--num-experts",type=int,default=32)
    p.add_argument("--top-k",type=int,default=2)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--output",type=Path,default=Path("artifacts/generation_benchmark.json"))
    a=p.parse_args()
    if a.new_tokens<1 or a.repeats<1 or a.warmup<0: p.error("invalid token/repeat/warmup count")
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA requested but unavailable")
    device=torch.device("cuda" if a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available()) else "cpu")
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type=="cuda" else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    patch=patch_model_with_checkpoints(model,a.checkpoint_dir,expected_layers=24,num_experts=a.num_experts,top_k=a.top_k)
    inputs=tokenizer(a.prompt,return_tensors="pt").to(device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    prefill_samples=[]; total_samples=[]; generated_counts=[]
    for run in range(a.warmup+a.repeats):
        sync(device); start=time.perf_counter()
        with torch.inference_mode():
            first=model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        sync(device); prefill=time.perf_counter()-start
        sync(device); start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=a.new_tokens,do_sample=False,use_cache=True,
                                  pad_token_id=tokenizer.eos_token_id)
        sync(device); total=time.perf_counter()-start
        if run>=a.warmup:
            prefill_samples.append(prefill); total_samples.append(total)
            generated_counts.append(max(0,int(output.shape[-1]-inputs["input_ids"].shape[-1])))
    median_total=statistics.median(total_samples)
    generated=statistics.median(generated_counts)
    report={"status":"benchmark_completed","model":a.model,"teacher_revision":revision,"device":str(device),"dtype":str(dtype),
            "prompt_tokens":int(inputs["input_ids"].shape[-1]),"requested_new_tokens":a.new_tokens,
            "actual_generated_tokens_median":generated,"repeats":a.repeats,"warmup":a.warmup,
            "prefill_plus_first_token_ms_median":round(statistics.median(prefill_samples)*1000,2),
            "full_generation_ms_median":round(median_total*1000,2),
            "decode_tokens_per_second_estimate":round(generated/max(median_total,1e-9),3),
            "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "patch_report":patch,
            "warning":"This measures the current Python reference dispatch on the selected device. It is not an Android result unless run on the target Android device."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
