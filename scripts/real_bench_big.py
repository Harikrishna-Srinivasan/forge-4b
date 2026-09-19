#!/usr/bin/env python3
"""Generalized bench: 110 items (MMLU-Pro 40 + GPQA 20 + GSM8K 30 + HumanEval 20).
Single-shot temp 0.2 (honest baseline); HumanEval uses std preamble (fixed harness).
Run detached: FORGE_ZEN=0 nohup python3 scripts/real_bench_big.py > evaluations/real-bench-big.log 2>&1 &
"""
import json, math, os, re, subprocess, tempfile, sys, time
def _tok():
    for line in open(os.path.expanduser("~/.env")):
        line=line.strip()
        if line.startswith("hftok="): return line.split("=",1)[1]
    return None
_t=_tok()
if _t: os.environ["HF_TOKEN"]=_t; os.environ["HUGGING_FACE_HUB_TOKEN"]=_t
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat
M=os.environ.get("FORGE_BENCH_MODEL","kamekichi128/qwen3-4b-instruct-2507:latest")
def ask(p, npred=64):
    for _ in range(2):
        try: return chat(M,[{"role":"system","content":"You are helpful. Be concise."},{"role":"user","content":p}],temperature=0.2,num_ctx=4096,num_predict=npred,think=False)
        except Exception: time.sleep(3)
    return "[failed]"
def ci(k,n):
    if not n: return ""
    p=k/n; se=math.sqrt(p*(1-p)/n); return f"±{1.96*se*100:.1f}pp"
def bench_mmlu(n=40):
    ds=load_dataset("TIGER-Lab/MMLU-Pro",split="test"); ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+j)}. {o}" for j,o in enumerate(ex["options"])])
        m=re.search(r"\b([A-J])\b",ask(f"{q}\n{opts}\nAnswer with single letter A-J only."))
        ok+=((m.group(1)==ex["answer"]) if m else 0); time.sleep(0.3)
    return ok,min(n,len(ds))
def bench_gpqa(n=20):
    ds=load_dataset("fingertap/GPQA-Diamond",split="test"); ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; gold=str(ex["answer"]).strip().upper()[:1]
        m=re.search(r"\b([A-D])\b",ask(f"{ex['question']}\nAnswer with single letter A-D only."))
        ok+=((m.group(1)==gold) if m else 0); time.sleep(0.3)
    return ok,min(n,len(ds))
def bench_gsm(n=30):
    try: ds=load_dataset("openai/gsm8k","main",split="test")
    except Exception: ds=load_dataset("gsm8k","main",split="test")
    ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; gold=re.search(r"####\s*(-?\d+)",ex["answer"]); gold=gold.group(1) if gold else "?"
        m=re.search(r"<a>\s*(-?\d+)",ask(f"{ex['question']}\nAnswer with number only inside <a> tags."))
        ok+=((m.group(1)==gold) if m else 0); time.sleep(0.3)
    return ok,min(n,len(ds))
def bench_he(n=20):
    try: ds=load_dataset("openai/openai_humaneval",split="test")
    except Exception: ds=load_dataset("openai_humaneval",split="test")
    ok=0; PRE="from typing import List, Dict, Tuple, Optional, Any\nimport math\n"
    for i in range(min(n,len(ds))):
        ex=ds[i]
        a=ask("Complete python:\n"+ex["prompt"]+"\nReturn ```python block only.",256)
        m=re.search(r"```python(.*?)```",a,re.S); code=(m.group(1) if m else a).strip()
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"s.py"); open(p,"w").write(PRE+code+"\n"+ex["test"])
            r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=10)
            ok+=(r.returncode==0)
        time.sleep(0.3)
    return ok,min(n,len(ds))
if __name__=="__main__":
    res={}
    for name,fn,n in [("MMLU-Pro",bench_mmlu,40),("GPQA-diamond",bench_gpqa,20),("GSM8K",bench_gsm,30),("HumanEval",bench_he,20)]:
        try:
            o,t=fn(n); res[name]={"score":f"{o}/{t} {o/t*100:.1f}%","ci95":ci(o,t)}
            print(f"{name}: {o}/{t} {o/t*100:.1f}% ({ci(o,t)})",flush=True)
        except Exception as e:
            res[name]={"score":f"error {e.__class__.__name__}"}; print(f"{name}: FAILED {e}",flush=True)
    tgt={"MMLU-Pro":85.2,"GPQA":86.0,"GSM8K":92.0,"HumanEval":73.4}
    json.dump({"model":M,"arch":"base-single-shot-fixed-harness","n":110,"results":res,"targets":tgt},
              open(os.environ.get("FORGE_BENCH_OUT","evaluations/real-bench-big.json"),"w"),indent=2)
    print(json.dumps(res,indent=2)); print("targets:",tgt)
