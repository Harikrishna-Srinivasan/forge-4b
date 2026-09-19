#!/usr/bin/env python3
"""Stratified MMLU-Pro: 3 items per category x 14 = 42 (MMLU-Pro test is category-ordered).
MODEL via FORGE_BENCH_MODEL, out via FORGE_BENCH_OUT (default evaluations/mmlu-strat.json).
Run detached, one model at a time (7GB RAM).
"""
import json, math, os, re, sys, time
def _tok():
    try:
        for line in open(os.path.expanduser("~/.env")):
            line=line.strip()
            if line.startswith("hftok="): return line.split("=",1)[1]
    except Exception: pass
    return None
_t=_tok()
if _t: os.environ["HF_TOKEN"]=_t; os.environ["HUGGING_FACE_HUB_TOKEN"]=_t
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat
M=os.environ.get("FORGE_BENCH_MODEL","kamekichi128/qwen3-4b-instruct-2507:latest")
OUT=os.environ.get("FORGE_BENCH_OUT","evaluations/mmlu-strat.json")
def ask(p):
    for _ in range(2):
        try: return chat(M,[{"role":"system","content":"You are helpful. Be concise."},{"role":"user","content":p}],temperature=0.2,num_ctx=4096,num_predict=64,think=False)
        except Exception: time.sleep(3)
    return "[failed]"
ds=load_dataset("TIGER-Lab/MMLU-Pro",split="test")
cats=sorted(set(ds[i]["category"] for i in range(len(ds))))
print("categories:",cats,flush=True)
idx={c:[] for c in cats}
for i in range(len(ds)): idx[ds[i]["category"]].append(i)
import random; random.seed(7)
pick=[]
for c in cats: pick+=random.sample(idx[c],3)
ok=0; per={}
for j,i in enumerate(pick):
    ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+k)}. {o}" for k,o in enumerate(ex["options"])])
    m=re.search(r"\b([A-J])\b",ask(f"{q}\n{opts}\nAnswer with single letter A-J only."))
    hit=(m.group(1)==ex["answer"]) if m else 0
    ok+=hit; per.setdefault(ex["category"],[0,0]); per[ex["category"]][0]+=hit; per[ex["category"]][1]+=1
    time.sleep(0.3)
n=len(pick); p=ok/n; se=math.sqrt(p*(1-p)/n)
print(f"STRAT-MMLU-Pro: {ok}/{n} {p*100:.1f}% (±{1.96*se*100:.1f}pp)",flush=True)
for c,(k,t) in sorted(per.items()): print(f"  {c}: {k}/{t}",flush=True)
json.dump({"model":M,"n":n,"score":f"{ok}/{n} {p*100:.1f}%","ci95":f"±{1.96*se*100:.1f}pp","per_category":{c:f"{k}/{t}" for c,(k,t) in per.items()}},open(OUT,"w"),indent=2)
