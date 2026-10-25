#!/usr/bin/env python3
"""Collect real Persian text from HF datasets using streaming; never synthesize fallback text."""
from __future__ import annotations
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [
    ("shekar-ai/Shiraz", "ODC-BY"),
    ("ErfanMoosaviMonazzah/mOSCAR-Persian", "CC-BY-4.0"),
]
def clean(value):
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count",type=int,default=64)
    p.add_argument("--min-chars",type=int,default=160)
    p.add_argument("--max-chars",type=int,default=4000)
    p.add_argument("--max-scan",type=int,default=12000)
    p.add_argument("--output",type=Path,default=Path("data/fa_persian_corpus.txt"))
    p.add_argument("--manifest-output",type=Path,default=Path("data/fa_persian_corpus.manifest.json"))
    a=p.parse_args()
    if min(a.count,a.min_chars,a.max_chars,a.max_scan)<1 or a.min_chars>a.max_chars:
        p.error("invalid positive limits")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install datasets first",file=sys.stderr)
        return 2
    errors=[]
    for source, license_name in SOURCES:
        rows=[]; provenance=[]
        try:
            print(f"Streaming real Persian corpus: {source}",flush=True)
            stream=load_dataset(source,split="train",streaming=True)
            for idx,row in enumerate(stream):
                if idx>=a.max_scan or len(rows)>=a.count:
                    break
                body=clean(row.get("text"))
                if len(body)<a.min_chars or row.get("judge_passed") is False:
                    continue
                rows.append(body[:a.max_chars])
                provenance.append({
                    "row_index":idx,
                    "url":str(row.get("url",""))[:1000],
                    "doc_id":str(row.get("doc_id",row.get("id","")))[:200]
                })
            if len(rows)<a.count:
                raise RuntimeError(f"only {len(rows)}/{a.count} valid rows after scanning {a.max_scan}")
        except Exception as exc:
            msg=f"{source}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print("WARNING: "+msg,file=sys.stderr,flush=True)
            continue
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.manifest_output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text("\n".join(rows)+"\n",encoding="utf-8")
        manifest={
            "status":"success","dataset":source,"split":"train","license":license_name,
            "streaming":True,"requested_samples":a.count,"collected_samples":len(rows),
            "min_chars":a.min_chars,"max_chars":a.max_chars,"max_scan":a.max_scan,
            "text_file":str(a.output),"collected_at_utc":datetime.now(timezone.utc).isoformat(),
            "fallback_used":source!=SOURCES[0][0],"documents":provenance,
            "note":"Real corpus text; no synthetic fallback. Preserve attribution and follow dataset license."
        }
        a.manifest_output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:manifest[k] for k in ("status","dataset","license","requested_samples","collected_samples","fallback_used")},ensure_ascii=False),flush=True)
        return 0
    print("ERROR: both real Persian datasets failed; refusing to train on synthetic fallback.\n"+"\n".join(errors),file=sys.stderr)
    return 1
if __name__=="__main__":
    raise SystemExit(main())
