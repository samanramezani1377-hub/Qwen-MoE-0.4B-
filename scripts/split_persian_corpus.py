#!/usr/bin/env python3
"""Deterministically split Persian documents before tokenization to prevent document leakage."""
import argparse, hashlib, json, random
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--train-output",type=Path,required=True)
    p.add_argument("--eval-output",type=Path,required=True)
    p.add_argument("--eval-fraction",type=float,default=0.15)
    p.add_argument("--seed",type=int,default=20261025)
    p.add_argument("--manifest-output",type=Path,required=True)
    a=p.parse_args()
    if not 0.05 <= a.eval_fraction <= 0.4: p.error("eval-fraction must be between 0.05 and 0.4")
    docs=[s.strip() for s in a.input.read_text(encoding="utf-8").splitlines() if s.strip()]
    unique={}
    for doc in docs: unique.setdefault(hashlib.sha256(doc.encode("utf-8")).hexdigest(),doc)
    pairs=[(doc,h) for h,doc in unique.items()]
    if len(pairs)<20: raise SystemExit(f"Need at least 20 unique documents; got {len(pairs)}")
    random.Random(a.seed).shuffle(pairs)
    n_eval=max(5,round(len(pairs)*a.eval_fraction)); n_eval=min(n_eval,len(pairs)-10)
    ev,tr=pairs[:n_eval],pairs[n_eval:]
    overlap={h for _,h in ev}&{h for _,h in tr}
    if overlap: raise RuntimeError("Document leakage detected")
    for path,subset in ((a.train_output,tr),(a.eval_output,ev)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("\n".join(d for d,_ in subset)+"\n",encoding="utf-8")
    manifest={"source_file":str(a.input),"seed":a.seed,"split_unit":"whole unique documents before tokenization",
      "train_documents":len(tr),"evaluation_documents":len(ev),
      "train_sha256":sorted(h for _,h in tr),"evaluation_sha256":sorted(h for _,h in ev),
      "overlap_documents":len(overlap),"leakage_check":"passed"}
    a.manifest_output.parent.mkdir(parents=True,exist_ok=True)
    a.manifest_output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if "sha256" not in k},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
